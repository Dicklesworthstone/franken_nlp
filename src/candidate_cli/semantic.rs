//! Explicit native INT8 extraction followed by correlated field diagnostics.
//! The separate command never changes ordinary extraction's acceptance policy.
use super::*;
use crate::{
    native_engine::{lmhead::scoring::ScoringLimits, strict_int8::prefill::Int8PrefillLimits},
    tasks::{extract::semantic::int8::Int8SemanticOptions, judge::JudgeLimits},
    tokenizer::untrusted::UntrustedEncoding,
};

pub(super) const OPTIONS_BYTES: usize = 1024 * 1024;

pub(super) mod corpus;
pub(super) mod jobs;

/// The complete-attempt encoding policy also has a separate invocation cap.
/// Actual work includes failed preparation and every dynamic review; neither
/// a record boundary nor a flush renews the mutable native processor ledger.
#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::candidate_cli) struct InvocationEncodingArgs {
    /// Total encoded input bytes across this invocation; safe-bpe-v1 only, default 67108864.
    #[arg(long)] max_total_tokenizer_input_bytes: Option<u64>,
    /// Total untrusted output tokens across this invocation; safe-bpe-v1 only, default 4194304.
    #[arg(long)] max_total_tokenizer_output_tokens: Option<u64>,
    /// Total BPE pair checks across this invocation; safe-bpe-v1 only, default 268435456.
    #[arg(long)] max_total_tokenizer_pair_checks: Option<u64>,
    /// Total examined BPE pair bytes across this invocation; safe-bpe-v1 only, default 4294967296.
    #[arg(long)] max_total_tokenizer_pair_bytes: Option<u64>,
}
impl InvocationEncodingArgs {
    pub(in crate::candidate_cli) fn limits(self, policy: &UntrustedEncoding)
        -> Result<Option<crate::tokenizer::untrusted::SafeBpeWorkLimits>, CandidateError> {
        policy.validate().map_err(|_| CandidateError::Arguments)?;
        if policy == &UntrustedEncoding::ByteFallback {
            return if self == Self::default() { Ok(None) } else { Err(CandidateError::Arguments) };
        }
        let limits = crate::tokenizer::untrusted::SafeBpeWorkLimits {
            max_input_bytes: self.max_total_tokenizer_input_bytes.unwrap_or(64 * MIB),
            max_output_tokens: self.max_total_tokenizer_output_tokens.unwrap_or(4 * MIB),
            max_pair_checks: self.max_total_tokenizer_pair_checks.unwrap_or(256 * MIB),
            max_pair_bytes: self.max_total_tokenizer_pair_bytes.unwrap_or(4096 * MIB),
        };
        limits.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(Some(limits))
    }
}

/// Population commands validate the exact fixed declaration without inventing
/// an empty document. Source-dependent membership is checked on each real item.
pub(in crate::candidate_cli) fn fixed_request(command: &SemanticCommand, schema: String,
    budget: crate::tasks::ir::TaskBudget, options: &Int8SemanticOptions)
    -> Result<crate::batch::extract::ExtractionBatchArgs, CandidateError> {
    let request = extract::arguments(schema, command.source_membership, budget)?;
    let declaration = crate::grammar::runtime::JsonValidationSchema::compile(
        &request.schema, extract::compiler(&command.host)).map_err(|_| CandidateError::Planning)?;
    if declaration.requires_verbatim_source() != command.source_membership {
        return Err(CandidateError::Planning);
    }
    options.validate_schema(declaration.root()).map_err(|_| CandidateError::Planning)?;
    Ok(request)
}

pub(in crate::candidate_cli) fn population_failure(error: CandidateError) -> stream::Failure {
    let (exit, code) = match error {
        CandidateError::Arguments => (ErrorCode::Usage, "semantic_population_options"),
        CandidateError::Input => (ErrorCode::InputDecodeOrParse, "semantic_population_input"),
        CandidateError::Planning => (ErrorCode::StructuredTaskNoResult, "semantic_population_plan"),
        CandidateError::Memory | CandidateError::Runtime =>
            (ErrorCode::AdmissionOrResourceLimit, "semantic_population_admission"),
        CandidateError::Identity => (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "semantic_population_identity"),
        CandidateError::Timeout => (ErrorCode::BudgetOrTimeout, "semantic_population_deadline"),
        _ => (ErrorCode::Generic, "semantic_population_setup_or_delivery"),
    };
    stream::Failure { exit, code, cancellation: None }
}

#[derive(Args)]
pub(crate) struct SemanticCommand {
    /// Exact original UTF-8 source; '-' or no path reads stdin.
    #[arg(default_value = "-")]
    pub(super) input: PathBuf,
    #[command(flatten)]
    pub(super) host: source::SourceHostArgs,
    /// Layer-major prompt morsels shared by extraction and every review head.
    /// Omitted keeps sequential prompt processing; valid rows are 1..=64.
    #[arg(long, value_name = "ROWS")]
    prefill_rows: Option<usize>,
    #[arg(skip)]
    pub(super) compute: compute::Args,
    /// Exact supported JSON schema in a bounded local file, never stdin.
    #[arg(long, value_name = "FILE")]
    pub(super) schema: PathBuf,
    /// Enforce verbatim source annotations during the original extraction.
    #[arg(long)]
    pub(super) source_membership: bool,
    /// Complete Int8SemanticOptions JSON with explicit experimental consent.
    #[arg(long, value_name = "FILE")]
    pub(super) semantic_options: PathBuf,
    #[arg(skip)]
    pub(super) encoding: input_encoding::DocumentInputEncodingArgs,
}

pub(super) fn definition() -> clap::Command {
    compute::definition(input_encoding::semantic_definition(SemanticCommand::augment_args(clap::Command::new("extract-semantic")
        .about("Extract exact-schema JSON and review configured scalar claims with the same INT8 model")
        .long_about("Run one real constrained extraction, then the existing full-source faithfulness judge for every configured scalar claim. --semantic-options is a complete Int8SemanticOptions JSON object with experimental_opt_in=true, versioned schema-aware claim rules, explicit judge policy/budget, semantic limits and all five whole-request work ceilings. All actual field judges compile and preflight before any review forward. The original maximum extraction reservation is never refunded after early EOS. The completed original JSON remains unchanged alongside entailed, contradicted, unsupported and not_checked diagnostics. Null, verbatim, unmapped and empty values remain explicitly not checked; absent optional properties are not invented. This is a same-model-correlated, uncalibrated diagnostic, not independent proof, automatic acceptance or field repair. One native context serves both stages with full-vocabulary projections. --prefill-rows explicitly groups bounded prompt morsels; --compute-threads admits one persistent row team across extraction and every review. Byte-fallback remains default; explicit safe-bpe-v1 shares one bounded tokenizer allowance across original extraction and every claim/evidence prompt. A failed stage produces no combined result. No sampling, retry, thinking or tools are accepted by this command."))))
        .mut_arg("compute_threads", |arg| arg.help("Coordinator plus persistent INT8 row workers, 1..=64; one team spans original extraction and all review heads, and composes with --prefill-rows. Default 1 keeps serial execution."))
        .mut_arg("max_result_bytes", |arg| arg.help("Original extraction result cap; combined result cap is semantic limits.max_output_bytes"))
        .after_long_help("The bounded options file is read first under the caller's preparation reservation. The complete declared source, retained field plans, native context, original extraction and combined output are then admitted before schema, source or model IO. The same deadline continues through planning, both native stages, serialization and flushing. Model-return alone does not release output ownership. These are modeled memory commitments and experimental diagnostics, with no quality or speed qualification.")
}

impl SemanticCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        command.encoding = input_encoding::DocumentInputEncodingArgs::from_matches(matches)?;
        command.compute = compute::Args::from_matches(matches)?;
        Ok(command)
    }

    pub(super) fn prefill(&self) -> Result<Option<Int8PrefillLimits>, CandidateError> {
        self.prefill_rows.map(|rows| {
            let bytes = Int8PrefillLimits::required_extra_scratch_bytes(rows)
                .map_err(|_| CandidateError::Arguments)?;
            Ok(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes })
        }).transpose()
    }

    pub(super) fn encoding_policy(&self) -> Result<UntrustedEncoding, CandidateError> {
        self.encoding.policy()
    }

    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        let prefill = self.prefill()?;
        let admission = self.compute.admit_prefill(prefill)?;
        extract::check_schema_path(&self.schema)?;
        extract::check_schema_path(&self.semantic_options)?;
        let (mut args, mut limits) = self.host.common(self.input.clone())?;
        args.compute = self.compute;
        limits.compute = admission;
        extract::validate_preparation(&self.host, &self.encoding_policy()?, limits)?;
        let encoding_workspace_bytes = self.encoding.workspace_bytes()?;
        // Options contain private claim templates. Price their parser/string
        // staging beside the exact schema, original source and output owner.
        let floor = (256 * MIB).checked_add(self.host.max_input_bytes as u64 * 32)
            .and_then(|n| n.checked_add(extract::SCHEMA_BYTES as u64 * 32))
            .and_then(|n| n.checked_add(self.host.max_result_bytes as u64 * 16))
            .and_then(|n| n.checked_add(OPTIONS_BYTES as u64 * 16))
            .and_then(|n| n.checked_add(encoding_workspace_bytes))
            .ok_or(CandidateError::Arguments)?;
        if floor > limits.preparation_bytes { return Err(CandidateError::Arguments); }
        Ok((args, limits))
    }

    pub(super) fn admit_options(&self, args: &CandidateArgs, limits: Limits,
        options: &Int8SemanticOptions) -> Result<usize, CandidateError> {
        options.validate().map_err(|_| CandidateError::Arguments)?;
        let judge = options.specification.judge_budget;
        let extraction = self.host.task_budget(limits);
        if judge.max_kv_bytes != limits.kv_bytes
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
        usize::try_from(options.limits.max_output_bytes).ok()
            .and_then(|n| n.checked_add(4096)).ok_or(CandidateError::Arguments)
    }

    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let (args, limits) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::semantic::execute(self, args, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

pub(super) fn parse_options(text: &str) -> Result<Int8SemanticOptions, CandidateError> {
    if text.len() > OPTIONS_BYTES { return Err(CandidateError::Input); }
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: 32, max_string_bytes: OPTIONS_BYTES,
    }).map_err(|_| CandidateError::Input)?;
    let options: Int8SemanticOptions = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    options.validate().map_err(|_| CandidateError::Arguments)?;
    Ok(options)
}

pub(super) fn judge_limits(options: &Int8SemanticOptions) -> JudgeLimits {
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

#[cfg(test)] mod tests;
