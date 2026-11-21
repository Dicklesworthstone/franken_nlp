//! Reconciled root-object extraction followed by complete-source field review.
use super::*;
use crate::{
    candidate_cli::stream::Failure,
    tasks::{judge::{FaithfulnessAggregationPolicy,
        grouped_int4::faithfulness_map::Int4FaithfulnessMapLimits}, mapreduce::ChunkLimits},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct SemanticObjectArgs {
    object: extract_map::object::ObjectMapArgs,
    semantic_options: PathBuf,
    review: ReviewArgs,
}
impl std::fmt::Debug for SemanticObjectArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SemanticObjectArgs")
    }
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct ReviewArgs {
    /// Maximum complete-source chunks for each scalar claim; excess refuses.
    #[arg(long, default_value_t = 64)]
    review_max_chunks: usize,
    #[arg(long)]
    review_max_chunk_bytes: Option<usize>,
    #[arg(long, default_value_t = 8192)]
    review_max_tokenizer_calls: usize,
    /// Retained evidence/results for one complete-source field review.
    #[arg(long, default_value_t = 16_777_216)]
    review_max_retained_result_bytes: u64,
    #[arg(long, default_value_t = 4_194_304)]
    review_max_result_bytes: u64,
    /// Minimum classified source coverage for the aggregate review policy.
    #[arg(long, default_value_t = 1_000_000)]
    review_minimum_classified_coverage_ppm: u32,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    ReviewArgs::augment_args(compute::definition(extract_map::object::definition()))
        .name("extract-semantic-map")
        .about("Reconcile a complete-document object and review scalar claims across its full source")
        .long_about("Extract complementary root-object properties across every lossless source chunk, reconcile exact values, then independently validate the final JSON against the ORIGINAL schema. Review configured non-verbatim scalar claims from that final object with the existing complete-source faithfulness map. Every field reviews the complete original document; no field is limited to its extraction chunk. The original map, reconciled JSON and exact source evidence remain unchanged. Every actual field/chunk/evidence-head plan is prepared and preflighted before the first review forward. All extraction and review passes share one five-axis native work allowance and one aggregate compact-tokenizer ledger. The semantic options' prompt and result limits cover all fields. Whole-source aggregate classifications retain contradictory, abstained and incomplete-coverage evidence; these are uncalibrated same-model-correlated diagnostics. No automatic acceptance, repair, omitted source windows or partial combined success. --compute-threads N keeps one admitted numerical team across extraction and review, joining all workers before completion. Source conversion and digest-pinned snapshots use the same real native path. Fixed policy and simultaneous memory admission precede schema/source/model IO. No selected-row projection, chunk cohorts, packed prefill, sampling, thinking, tools or durable resume on this route.")
        .arg(clap::Arg::new("semantic_options").long("semantic-options").required(true)
            .value_name("FILE").value_parser(clap::value_parser!(PathBuf))
            .help("Complete explicit Int4SemanticOptions JSON; max_model_work bounds extraction plus ALL complete-source reviews"))
        .mut_arg("max_selected_rows", |arg| arg.hide(true))
        .mut_arg("cohort_chunks", |arg| arg.hide(true))
        .mut_arg("max_map_result_bytes", |arg| arg.default_value("4194304")
            .help("Original extraction-map-plus-object cap; combined result uses semantic limits.max_output_bytes"))
        .mut_arg("max_object_json_bytes", |arg| arg.default_value("1048576"))
        .after_long_help("--input-encoding safe-bpe-v1 uses ONE nonrenewable ledger for original extraction, chunk probes, field claims and every review head. Byte fallback is the default. Per-review chunk/result limits bound each field; semantic max_fields, max_total_prompt_tokens, max_total_projected_logits, max_output_bytes and max_model_work bound the entire invocation. Optional snapshot save occurs after fixed-policy/schema and memory admission; later source-dependent extraction/review may still refuse.")
}

impl SemanticObjectArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<Self, clap::Error> {
        let semantic_options = matches.get_one::<PathBuf>("semantic_options").cloned().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingRequiredArgument, "semantic options file required"))?;
        Ok(Self { object: extract_map::object::ObjectMapArgs::from_matches(matches)?,
            semantic_options, review: ReviewArgs::from_arg_matches(matches)? })
    }
    fn validate(&self, args: &Int4Args)
        -> Result<extract_map::object::ObjectMapLimits, CandidateError> {
        crate::candidate_cli::extract::check_schema_path(&self.semantic_options)?;
        self.review.validate()?;
        self.object.validate_semantic(args)
    }
    fn review_limits(&self, args: &Int4Args, common: Int4Limits, options: &Int4SemanticOptions)
        -> Result<Int4FaithfulnessMapLimits, CandidateError> {
        options.validate().map_err(|_| CandidateError::Arguments)?;
        self.review.validate()?;
        let budget = options.specification.judge_budget;
        if budget.max_kv_bytes < common.native.kv_bytes
            || budget.max_input_tokens as usize > args.context_tokens
            || budget.max_output_tokens == 0 || budget.max_output_tokens > 1024
            || budget.max_input_tokens.checked_add(budget.max_output_tokens)
                .and_then(|tokens| tokens.checked_sub(1)).is_none_or(|tokens| tokens as usize > args.context_tokens)
            || !(1..=65_536).contains(&budget.max_grammar_states)
            || budget.max_output_bytes > options.limits.max_output_bytes
            || self.review.review_max_result_bytes > options.limits.max_output_bytes
            || budget.max_output_bytes > self.review.review_max_retained_result_bytes {
            return Err(CandidateError::Arguments);
        }
        let chunks = ChunkLimits {
            max_input_bytes: args.max_input_bytes.checked_add(options.limits.max_claim_bytes)
                .ok_or(CandidateError::Arguments)?,
            max_chunk_bytes: self.review.review_max_chunk_bytes.unwrap_or(args.max_input_bytes),
            max_chunk_tokens: budget.max_input_tokens as usize,
            context_tokens: args.context_tokens, reserved_tokens: budget.max_output_tokens as usize,
            max_chunks: self.review.review_max_chunks,
            max_tokenizer_calls: self.review.review_max_tokenizer_calls,
        };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let aggregation = FaithfulnessAggregationPolicy {
            minimum_classified_coverage_ppm: self.review.review_minimum_classified_coverage_ppm,
        };
        aggregation.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(Int4FaithfulnessMapLimits { chunks, max_model_work: options.max_model_work,
            max_retained_result_bytes: self.review.review_max_retained_result_bytes,
            max_result_bytes: self.review.review_max_result_bytes, aggregation })
    }
    pub(in crate::candidate_cli::int4) fn run(self, args: Int4Args, input: &mut impl Read,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let _ = writeln!(diagnostics, "fnlp candidate semantic object map: {}", failure.code);
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), Failure> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output);
            Err(Failure::usage("semantic_object_map_profile_unavailable")) }
    }
}
impl ReviewArgs {
    fn validate(self) -> Result<(), CandidateError> {
        if !(1..=256).contains(&self.review_max_chunks)
            || !(1..=1_000_000).contains(&self.review_max_tokenizer_calls)
            || self.review_max_chunk_bytes.is_some_and(|bytes| !(4..=MAX_INPUT_BYTES).contains(&bytes))
            || !(1..=64 * MIB).contains(&self.review_max_retained_result_bytes)
            || !(1..=64 * MIB).contains(&self.review_max_result_bytes)
            || self.review_minimum_classified_coverage_ppm > 1_000_000 {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
}
