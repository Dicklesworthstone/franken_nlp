//! Explicit summary/QA generation plus an unchanged correlated second reader.
use super::*;
use crate::{
    candidate_cli::source::Kind,
    native_engine::lmhead::scoring::ScoringLimits,
    tasks::{judge::JudgeLimits,
        source_planning::grouped_int4::semantic::{Int4SourceSemanticOptions, SourceSemanticInputFacts}},
};

const OPTIONS_BYTES: usize = 1024 * 1024;
#[cfg(feature = "asupersync-runtime")] mod runtime;
#[cfg(test)] mod tests;

#[derive(Clone, Eq, PartialEq)]
pub(super) struct SemanticSourceArgs {
    source: source::SourceArgs,
    semantic_options: PathBuf,
    input_encoding: input_encoding::DocumentInputEncodingArgs,
}
impl std::fmt::Debug for SemanticSourceArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("SemanticSourceArgs") }
}
pub(super) fn definitions() -> Vec<clap::Command> {
    [(Kind::Summarize, "summarize-semantic"), (Kind::Answer, "answer-semantic")].into_iter().map(|(kind, name)| {
        input_encoding::semantic_definition(compute::definition(source::definition(kind))).name(name)
            .about("Generate a cited source result and explicitly review it against the complete original evidence")
            .long_about("Run the existing cited-summary or passage-answer task, then the native faithfulness judge for each complete summary bullet or the complete question/answer pair. --semantic-options supplies a complete Int4SourceSemanticOptions JSON with experimental_opt_in=true, explicit judge policy/budget, statement limits and five-axis whole-request work ceilings. The original source result, generated tokens, citations and semantic_support=not_assessed remain unchanged. Separate entailed/contradicted/unsupported/judge_abstained/not_checked diagnostics are uncalibrated and same-model correlated. Review sees the COMPLETE original document or all original passages, not just the displayed citations; it does not certify citation-only entailment or atomize every material claim. Question and passage IDs are not evidence. Question/answer claim rendering and the exact passage layout are versioned and bound to the original request. An abstained answer or empty summary is explicitly not_checked; failed decoding, review admission, cancellation or native execution never becomes abstention. All dynamic judgments must compile and preflight before any review forward. One native engine and one nonrenewable source-reservation-plus-review work allowance cover the complete result. --compute-threads N keeps one admitted numerical team across source generation and every complete-statement judgment. Worker scratch is quoted before private input/model IO; separate physical counters describe actual projections and all workers join before completion. Source conversion and independently pinned snapshots share the route. Optional snapshot saving follows fixed-policy, evidence and memory admission. --input-encoding safe-bpe-v1 shares one bounded tokenizer allowance across source/question/passage preparation and all review claims/evidence windows. No repair, retry, selected output rows, prompt prefill, sampling, thinking, tools or durable resume. Blocking IO observes one end-to-end deadline cooperatively.")
            .arg(clap::Arg::new("semantic_options").long("semantic-options").required(true)
                .value_name("FILE").value_parser(clap::value_parser!(PathBuf))
                .help("Complete explicit Int4SourceSemanticOptions JSON; bounded local file, never stdin"))
            .mut_arg("max_result_bytes", |arg| arg.help("Original source result cap; combined output uses semantic limits.max_output_bytes"))
    }).collect()
}
impl SemanticSourceArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let source = source::SourceArgs::from_matches(matches)?;
        let semantic_options = matches.get_one::<PathBuf>("semantic_options").cloned().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingRequiredArgument, "semantic options file required"))?;
        Ok(Self { source, semantic_options,
            input_encoding: input_encoding::DocumentInputEncodingArgs::from_matches(matches)? })
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        crate::candidate_cli::extract::check_schema_path(&self.semantic_options)?;
        let remote = |path: &std::path::Path| path.to_str().is_some_and(|text| text.contains("://"));
        if remote(&self.semantic_options) || self.source.options_path().is_some_and(remote) {
            return Err(CandidateError::Arguments);
        }
        if args.prefill()?.is_some() || !args.input_encoding.is_default() {
            return Err(CandidateError::Arguments);
        }
        let common = self.source.validate_batch_with_prefill(args)?;
        self.storage(args, common, common.result_bytes)
    }
    fn storage(&self, args: &Int4Args, original: Int4Limits, result_bytes: usize)
        -> Result<Int4Limits, CandidateError> {
        let source = self.source.budget(args, original);
        let extra = sum(&[128 * MIB, crate::candidate_cli::source::OPTIONS_BYTES as u64 * 16,
            source.max_output_bytes * 16, u64::from(source.max_grammar_states) * 32, OPTIONS_BYTES as u64 * 16,
            self.input_encoding.workspace_bytes()?])?;
        args.validate_source_storage(result_bytes, args.max_new_tokens, original.max_prompt_tokens, extra)
    }
    fn admit_options(&self, args: &Int4Args, common: Int4Limits, options: &Int4SourceSemanticOptions)
        -> Result<Int4Limits, CandidateError> {
        options.validate().map_err(|_| CandidateError::Arguments)?;
        let judge = options.judge_budget; let source = self.source.budget(args, common);
        if judge.max_kv_bytes != common.native.kv_bytes || judge.max_input_tokens as usize > args.context_tokens
            || judge.max_output_tokens == 0 || judge.max_output_tokens > 1024
            || judge.max_input_tokens.checked_add(judge.max_output_tokens).and_then(|n| n.checked_sub(1))
                .is_none_or(|n| n as usize > args.context_tokens)
            || !(1..=65_536).contains(&judge.max_grammar_states)
            || source.max_output_bytes > options.limits.max_output_bytes {
            return Err(CandidateError::Arguments);
        }
        self.storage(args, common, usize::try_from(options.limits.max_output_bytes).map_err(|_| CandidateError::Arguments)?)
    }
    pub(super) fn execute(self, kind: Kind, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        if !matches!(kind, Kind::Summarize | Kind::Answer) { return Err(CandidateError::Arguments); }
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(kind, args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, kind, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
fn parse_options(text: &str) -> Result<Int4SourceSemanticOptions, CandidateError> {
    if text.len() > OPTIONS_BYTES { return Err(CandidateError::Input); }
    let value = crate::canonjson::parse_str(text).map_err(|_| CandidateError::Input)?;
    let options: Int4SourceSemanticOptions = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    options.validate().map_err(|_| CandidateError::Arguments)?;
    Ok(options)
}
fn judge_limits(options: &Int4SourceSemanticOptions) -> JudgeLimits {
    let budget = options.judge_budget; let nodes = budget.max_grammar_states as usize;
    let logits = options.limits.max_total_projected_logits.min(options.max_model_work.projected_logits);
    JudgeLimits { per_head: ScoringLimits { max_candidates: 3, max_total_tokens: nodes.saturating_sub(1),
        max_nodes: nodes, max_depth: budget.max_output_tokens as usize, max_candidate_id_bytes: 64,
        max_projected_logits: logits }, max_total_prompt_tokens: options.limits.max_total_prompt_tokens,
        max_total_candidate_tokens: nodes, max_total_projected_logits: logits, max_output_bytes: budget.max_output_bytes }
}
fn input_ceiling(kind: Kind, args: &Int4Args, options: &Int4SourceSemanticOptions)
    -> Result<SourceSemanticInputFacts, CandidateError> {
    let source_bytes = args.max_input_bytes.min(options.limits.max_source_bytes);
    Ok(match kind {
        Kind::Summarize => SourceSemanticInputFacts { source_bytes, question_bytes: 0, passage_count: 0 },
        Kind::Answer => SourceSemanticInputFacts { source_bytes,
            question_bytes: args.max_input_bytes.min(options.limits.max_claim_bytes),
            passage_count: crate::candidate_cli::source::MAX_PASSAGES },
        _ => return Err(CandidateError::Arguments),
    })
}
