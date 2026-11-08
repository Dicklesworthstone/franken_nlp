//! Exact-schema extraction over the shared explicit INT4 source/snapshot path.
//! The schema remains raw UTF-8; no serde Value round trip can round decimals.

use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::{JsonProgram, SourceRuntimeLimits}},
    native_engine::constrained_int4::sparse::Int4JsonSparseLimits,
    tasks::ir::TaskBudget,
};

const TEXT_ONLY: &[&str] = &[
    "max_output_bytes", "seed", "temperature_milli", "top_k", "top_p_ppm", "logprobs",
    "prefill_rows", "min_new_tokens", "stop_suffixes", "banned_token_ids", "repetition_penalty_milli",
    "presence_penalty_milli", "frequency_penalty_milli", "logit_bias",
];

#[derive(Args, Clone, Eq, PartialEq)]
pub(super) struct ExtractionArgs {
    /// Bounded local JSON schema file. '-' and remote URLs are not schema sources.
    #[arg(long, value_name = "FILE")]
    schema: PathBuf,
    /// Enforce verbatim schema fields against the exact input bytes.
    /// Requires at least one x-fnlp-source=verbatim field; not semantic verification.
    #[arg(long)]
    source_membership: bool,
    /// Complete typed extraction result, including source evidence and work.
    #[arg(long, default_value_t = 1_048_576)]
    max_result_bytes: usize,
    #[arg(long, default_value_t = 4096)]
    max_grammar_states: u32,
    /// Conservative aggregate mask-traversal work, including the EOS choice.
    #[arg(long, default_value_t = 2_000_000_000)]
    max_mask_node_visits: u64,
    #[arg(long, default_value_t = 2_000_000)]
    mask_step_node_visits: u64,
    // Only the one-off command installs this argument. Shared definitions
    // cannot accidentally expose an execution strategy they do not implement.
    #[arg(skip)]
    max_selected_rows: Option<usize>,
}
impl std::fmt::Debug for ExtractionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("ExtractionArgs") }
}

pub(super) fn definition() -> clap::Command {
    let mut command = Int4Args::augment_args(clap::Command::new("extract")
        .about("Extract schema-valid JSON and optional exact source fields using grouped INT4")
        .long_about("Compile a bounded local schema against the exact UTF-8 document before model access. Execute full-vocabulary greedy grammar decoding on the explicitly selected source-converted or snapshot-loaded INT4 model, then independently validate the complete result. --source-membership binds every annotated verbatim field to the original document and retains all matching offsets. JSON is returned as exact text inside the result envelope. Schema validity and source membership do not certify semantic accuracy. Greedy constrained decoding has no sampling, repair retry or sparse projection. Prompt execution is sequential by default; only the one-off extract command exposes --prefill-rows."))
        .mut_arg("input", |arg| arg.help("Exact UTF-8 source document; '-' reads stdin"))
        .mut_arg("max_new_tokens", |arg| arg.default_value("512").help("Constrained output tokens INCLUDING terminal EOS"))
        .mut_arg("preparation_mib", |arg| arg.default_value("512"));
    for &id in TEXT_ONLY { command = command.mut_arg(id, |arg| arg.hide(true)); }
    ExtractionArgs::augment_args(command)
}

pub(super) fn one_off_definition() -> clap::Command {
    definition()
        .long_about("Compile a bounded local schema against the exact UTF-8 document before model access. Full-vocabulary greedy grammar decoding is the default. Explicit --max-selected-rows N selects every legal grammar row and refuses a step whose complete legal set exceeds N, including EOS when accepting. This is a bounded head projection strategy, never top-k pruning; it checks requested logits and does not certify parity with unrequested full-head rows. The strategy and exact row ceiling enter execution identity before hosted admission. --source-membership preserves all original occurrences of annotated verbatim fields. JSON remains exact text and the complete result is independently validated. Schema validity and source membership do not certify semantic accuracy. No sampling or repair retry. --prefill-rows independently selects bounded layer-major prompt execution. Batch streams expose their own fixed selected-row option, including document cohorts and packed prompt passes. Complete-document maps, corpora and retained document jobs accept a fixed selected-row policy for all chunks. Single-context extract-job and semantic workflows retain full-vocabulary execution.")
        .arg(selected_rows_argument())
}
pub(super) fn selected_rows_argument() -> clap::Arg {
    clap::Arg::new("max_selected_rows").long("max-selected-rows")
        .value_name("N").value_parser(clap::value_parser!(usize))
        .help("Opt into all legal grammar-row projections; 1..=166144 rows per step including accepted EOS; excess refuses")
}
impl ExtractionArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let args = Self::from_matches_with_projection(matches)?;
        if args.max_selected_rows.is_some() {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "this INT4 extraction workflow requires full-vocabulary head projection"));
        }
        Ok(args)
    }
    /// A composed map may select a fixed head policy while keeping ordinary
    /// sequential prompt execution. Population routes validate their own scope.
    pub(super) fn from_matches_with_projection(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if matches.value_source("prefill_rows") == Some(clap::parser::ValueSource::CommandLine) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "this INT4 extraction workflow requires sequential prompt execution"));
        }
        Self::from_matches_with_prefill(matches)
    }
    pub(super) fn from_matches_with_prefill(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if TEXT_ONLY.iter().filter(|&&id| id != "prefill_rows")
            .any(|id| matches.value_source(id) == Some(clap::parser::ValueSource::CommandLine)) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "INT4 extraction accepts constrained tokens and result limits, not generation controls"));
        }
        let mut args = Self::from_arg_matches(matches)?;
        args.max_selected_rows = matches.try_get_one::<usize>("max_selected_rows").ok().flatten().copied();
        Ok(args)
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        if args.prefill()?.is_some() { return Err(CandidateError::Arguments); }
        self.validate_full_projection_with_prefill(args)
    }
    pub(super) fn validate_with_projection(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        if args.prefill()?.is_some() { return Err(CandidateError::Arguments); }
        self.validate_with_prefill(args)
    }
    /// Composed streams own their projection policy separately from one-off
    /// extraction arguments, even when they expose prompt scheduling.
    pub(super) fn validate_full_projection_with_prefill(&self, args: &Int4Args)
        -> Result<Int4Limits, CandidateError> {
        if self.max_selected_rows.is_some() { return Err(CandidateError::Arguments); }
        self.validate_with_prefill(args)
    }
    pub(super) fn validate_with_prefill(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        // Composed workflows still rely on byte-based chunk capacity. Their
        // shared argument type must not enable a policy they never bind.
        if !args.input_encoding.is_default() || args.compute.threads() != 1 { return Err(CandidateError::Arguments); }
        self.validate_one_off_with_prefill(args)
    }
    /// Resident streams own fixed output-row and input-encoding policies and
    /// may install one persistent team. Maps/jobs keep their byte-only guard.
    pub(super) fn validate_batch_with_prefill(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        if self.max_selected_rows.is_some() {
            return Err(CandidateError::Arguments);
        }
        self.validate_one_off_with_prefill(args)
    }
    pub(super) fn validate_one_off_with_prefill(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        let _ = self.selected_rows()?;
        crate::candidate_cli::extract::check_schema_path(&self.schema)?;
        if !(1..=MAX_CONTENT_BYTES).contains(&self.max_result_bytes)
            || !(1..=65_536).contains(&self.max_grammar_states)
            || args.max_new_tokens == 0 || args.max_new_tokens > 1024 || args.max_new_tokens >= args.context_tokens
            || !(1..=1_000_000_000).contains(&self.mask_step_node_visits)
            || !(1..=1_000_000_000_000).contains(&self.max_mask_node_visits)
            || args.max_output_bytes != 65_536
            || args.options(166_101)? != GenerationOptions::greedy(args.max_new_tokens, 65_536, 166_101) {
            return Err(CandidateError::Arguments);
        }
        let mask_bound = self.mask_step_node_visits.checked_mul(args.max_new_tokens as u64)
            .ok_or(CandidateError::Arguments)?;
        if mask_bound > self.max_mask_node_visits { return Err(CandidateError::Arguments); }
        // Covers the pinned vocabulary trie, bounded source-index preparation,
        // schema/IR copies and complete source-evidence/output staging. These
        // are declared reserves, not an operating-system RSS guarantee.
        let max_prompt_tokens = args.context_tokens - args.max_new_tokens + 1;
        let encoding_extra = self.encoding_workspace_extra(args, max_prompt_tokens)?;
        let extra = sum(&[128 * MIB, crate::candidate_cli::extract::SCHEMA_BYTES as u64 * 16,
            self.max_result_bytes as u64 * 16, u64::from(self.max_grammar_states) * 32, encoding_extra])?;
        args.validate_source_storage(self.max_result_bytes, args.max_new_tokens,
            max_prompt_tokens, extra)
    }
    /// The exact schema and document are two untrusted segments in one prompt.
    /// Their aggregate cap cannot exceed the tokenizer's complete-request
    /// ceiling, even when the separately declared document cap is that large.
    pub(super) fn encoding_policy(&self, args: &Int4Args, max_prompt_tokens: usize)
        -> Result<crate::tokenizer::untrusted::UntrustedEncoding, CandidateError> {
        let bytes = args.max_input_bytes.checked_add(crate::candidate_cli::extract::SCHEMA_BYTES)
            .ok_or(CandidateError::Arguments)?
            .min(crate::tokenizer::untrusted::SafeBpeLimits::MAX_INPUT_BYTES);
        args.input_encoding.policy(bytes, max_prompt_tokens)
    }
    fn encoding_workspace_extra(&self, args: &Int4Args, max_prompt_tokens: usize)
        -> Result<u64, CandidateError> {
        use crate::tokenizer::untrusted::UntrustedEncoding;
        let complete = match self.encoding_policy(args, max_prompt_tokens)? {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(policy) => policy.requirements()
                .map_err(|_| CandidateError::Arguments)?.workspace_bytes,
        };
        // Common admission already prices the document-only quote. Charge
        // the remaining schema allowance exactly once, before either file IO.
        complete.checked_sub(args.input_encoding.workspace_bytes(args.max_input_bytes, max_prompt_tokens)?)
            .ok_or(CandidateError::Arguments)
    }
    pub(super) fn budget(&self, args: &Int4Args, limits: Int4Limits) -> TaskBudget {
        TaskBudget { max_input_tokens: limits.max_prompt_tokens as u32,
            max_output_tokens: args.max_new_tokens as u32, max_output_bytes: self.max_result_bytes as u64,
            max_grammar_states: self.max_grammar_states, max_kv_bytes: limits.native.kv_bytes }
    }
    pub(super) fn compiler(&self) -> CompileLimits {
        CompileLimits { max_schema_bytes: crate::candidate_cli::extract::SCHEMA_BYTES,
            max_output_bytes: self.max_result_bytes, max_states: self.max_grammar_states as usize,
            ..CompileLimits::default() }
    }
    pub(super) fn source_limits(&self) -> SourceRuntimeLimits { SourceRuntimeLimits::default() }
    pub(super) fn masks(&self) -> MaskWorkLimits {
        MaskWorkLimits { max_trie_node_visits: self.mask_step_node_visits as usize, checkpoint_interval_nodes: 256 }
    }
    pub(super) fn mask_visits(&self) -> u64 { self.max_mask_node_visits }
    pub(super) fn schema_path(&self) -> &std::path::Path { &self.schema }
    pub(super) fn selected_rows(&self) -> Result<Option<Int4JsonSparseLimits>, CandidateError> {
        self.max_selected_rows.map(|max_rows_per_step| {
            let limits = Int4JsonSparseLimits { max_rows_per_step };
            limits.validate().map_err(|_| CandidateError::Arguments)?;
            Ok(limits)
        }).transpose()
    }
    /// Seal a document-wide head policy before preparing any real chunk. The
    /// same immutable planner is used for preflight and hosted re-preparation.
    #[cfg(any(feature = "asupersync-runtime", test))]
    pub(super) fn configure_planner(&self, planner: crate::batch::extract::grouped_int4::Int4ExtractionBatchPlanner)
        -> Result<crate::batch::extract::grouped_int4::Int4ExtractionBatchPlanner, CandidateError> {
        if planner.selected_rows().is_some() { return Err(CandidateError::Planning); }
        match self.selected_rows()? {
            Some(limits) => planner.with_selected_rows(limits).map_err(|_| CandidateError::Planning),
            None => Ok(planner),
        }
    }
    /// Bind the explicit head strategy before the host sees an admitted plan.
    #[cfg(any(feature = "asupersync-runtime", test))]
    pub(super) fn configure_plan(&self, plan: crate::tasks::extract::grouped_int4::Int4ExtractPlan)
        -> Result<crate::tasks::extract::grouped_int4::Int4ExtractPlan, CandidateError> {
        if plan.selected_rows().is_some() { return Err(CandidateError::Planning); }
        match self.selected_rows()? {
            Some(limits) => plan.with_selected_rows(limits).map_err(|_| CandidateError::Planning),
            None => Ok(plan),
        }
    }
    #[cfg(test)]
    pub(super) fn source_membership(&self) -> bool { self.source_membership }

    /// Preserve exact shared schema bytes without inventing a source document.
    /// Source membership compiles against EACH original job/corpus document.
    pub(super) fn shared_defaults(&self, schema: String, ceiling: TaskBudget)
        -> Result<ExtractionBatchArgs, CandidateError> {
        if self.max_selected_rows.is_some() { return Err(CandidateError::Arguments); }
        self.shared_defaults_with_projection(schema, ceiling)
    }
    /// Projection policy belongs to the planner, never the per-document schema
    /// or budget. Only workflows that bind that planner may use this helper.
    pub(super) fn shared_defaults_with_projection(&self, schema: String, ceiling: TaskBudget)
        -> Result<ExtractionBatchArgs, CandidateError> {
        let _ = self.selected_rows()?;
        ceiling.validate().map_err(|_| CandidateError::Planning)?;
        let request = crate::candidate_cli::extract::arguments(schema, self.source_membership, ceiling)?;
        let declaration = crate::validation::parse_json_with_limits(&request.schema, crate::validation::JsonLimits {
            max_input_bytes: crate::candidate_cli::extract::SCHEMA_BYTES,
            max_string_lexeme_bytes: crate::candidate_cli::extract::SCHEMA_BYTES,
            ..crate::validation::JsonLimits::default()
        }).map_err(|_| CandidateError::Planning)?;
        if !matches!(declaration, crate::validation::JsonValue::Object(_)) {
            return Err(CandidateError::Planning);
        }
        if request.grounding == ExtractionBatchGrounding::Structural {
            JsonProgram::compile(&request.schema, self.compiler()).map_err(|_| CandidateError::Planning)?;
        }
        Ok(request)
    }

    /// Reject unsupported/duplicate schema keys and impossible source products
    /// before any model file is opened. Keep exact numbers as original bytes.
    pub(super) fn check_schema(&self, schema: String, document: &str, budget: TaskBudget)
        -> Result<ExtractionBatchArgs, CandidateError> {
        let request = crate::candidate_cli::extract::arguments(schema, self.source_membership, budget)?;
        let program = match request.grounding {
            ExtractionBatchGrounding::Structural => JsonProgram::compile(&request.schema, self.compiler()),
            ExtractionBatchGrounding::SourceMembership => JsonProgram::compile_with_source(
                &request.schema, document, self.compiler(), self.source_limits()),
        }.map_err(|_| CandidateError::Planning)?;
        if self.source_membership != program.requires_source() { return Err(CandidateError::Planning); }
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod selected;
    mod encoding;
    fn command(extra: &[&str]) -> Int4Command {
        let mut argv = vec!["candidate", "source-int4", "extract", "--source-dir", "explicit-source",
            "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192", "--schema", "schema.json"];
        argv.extend_from_slice(extra);
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong dispatcher"); };
        command
    }
    fn arguments(extra: &[&str]) -> (ExtractionArgs, Int4Args) {
        let command = command(extra);
        let Int4Task::Extract(extract) = command.task else { panic!("wrong task"); };
        (extract, command.args)
    }
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid extraction reached private input"); }
    }
    #[test]
    fn source_and_pinned_snapshot_routes_share_extraction_without_an_int8_alias() {
        let (extract, args) = arguments(&[]);
        let limits = extract.validate(&args).unwrap();
        assert_eq!(args.max_new_tokens, 512);
        assert!(limits.prefill.is_none());
        assert_eq!(extract.budget(&args, limits).max_output_bytes, 1_048_576);
        assert_eq!(limits.max_prompt_tokens + args.max_new_tokens - 1, args.context_tokens);
        let digest = "ab".repeat(32);
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", "extract", "--snapshot", "local.snapshot",
            "--expected-model-sha256", &digest, "--memory-mib", "8192", "--schema", "schema.json",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong snapshot route"); };
        let Int4Task::Extract(extract) = command.task else { panic!("wrong snapshot task"); };
        assert!(extract.validate(&command.args).unwrap().snapshot_pin.is_some());
    }
    #[test]
    fn generation_controls_are_rejected_even_when_explicit_defaults() {
        let seed = "ab".repeat(32);
        for flags in [vec!["--seed", seed.as_str()], vec!["--max-output-bytes", "65536"],
            vec!["--logprobs"], vec!["--min-new-tokens", "0"],
            vec!["--stop", "END"], vec!["--ban-token", "7"], vec!["--repetition-penalty-milli", "1000"],
            vec!["--presence-penalty-milli", "0"], vec!["--frequency-penalty-milli", "0"],
            vec!["--logit-bias", "42=0"]] {
            let mut argv = vec!["candidate", "source-int4", "extract", "--source-dir", "source",
                "--source-manifest", "manifest", "--memory-mib", "8192", "--schema", "schema.json"];
            argv.extend(flags);
            let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
            assert!(CandidateCommand::from_matches(&matches).is_err());
        }
    }
    #[test]
    fn one_off_prefill_preserves_schema_grounding_and_task_budget_but_shared_validation_stays_strict() {
        let (serial, args) = arguments(&["--source-membership"]);
        let original = serial.validate(&args).unwrap();
        let (grouped, args) = arguments(&["--source-membership", "--prefill-rows", "4"]);
        let limits = grouped.validate_with_prefill(&args).unwrap();
        assert_eq!(serial, grouped);
        let budget = grouped.budget(&args, limits);
        assert_eq!(budget, serial.budget(&args, original));
        assert_eq!(grouped.mask_visits(), serial.mask_visits());
        assert_eq!(limits.prefill.unwrap().max_batch_rows, 4);
        assert_eq!(grouped.validate(&args).unwrap_err(), CandidateError::Arguments);
        let schema = " {\"type\":\"string\",\"const\":\"上海\",\"x-fnlp-source\":\"verbatim\"}\n";
        let request = grouped.check_schema(schema.to_owned(), "上海 上海", budget).unwrap();
        assert_eq!(request.schema, schema);
        assert_eq!(request.grounding, ExtractionBatchGrounding::SourceMembership);
        assert!(grouped.check_schema(schema.to_owned(), "Beijing", budget).is_err());
    }
    #[test]
    fn bad_budgets_or_programmatic_sampler_policy_refuse_before_io() {
        for flags in [vec!["--max-result-bytes", "0"], vec!["--max-result-bytes", "1048577"],
            vec!["--max-grammar-states", "0"], vec!["--max-grammar-states", "65537"],
            vec!["--max-new-tokens", "0"], vec!["--max-new-tokens", "2048"],
            vec!["--mask-step-node-visits", "0"], vec!["--max-mask-node-visits", "1"],
            vec!["--max-mask-node-visits", "1000000000001"], vec!["--preparation-mib", "1"]] {
            let mut output = Vec::new();
            assert_eq!(command(&flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
            assert!(output.is_empty());
        }
        let mut command = command(&[]); command.args.seed = Some("ab".repeat(32));
        assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
    #[test]
    fn grammar_preflight_preserves_exact_decimals_and_original_source_bytes() {
        let (extract, args) = arguments(&[]);
        let limits = extract.validate(&args).unwrap();
        let schema = r#"{"type":"number","const":12345678901234567890123456789012345678}"#;
        let parsed = extract.check_schema(schema.to_owned(), "source", extract.budget(&args, limits)).unwrap();
        assert_eq!(parsed.schema, schema);
        let (grounded, args) = arguments(&["--source-membership"]);
        let budget = grounded.budget(&args, grounded.validate(&args).unwrap());
        let schema = r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#;
        assert!(grounded.check_schema(schema.to_owned(), "上海 上海", budget).is_ok());
        assert!(grounded.check_schema(schema.to_owned(), "Beijing", budget).is_err());
        assert!(extract.check_schema(schema.to_owned(), "上海", budget).is_err());
    }
    #[test]
    fn unsupported_duplicate_or_unbound_grounding_schemas_refuse_during_preflight() {
        let (extract, args) = arguments(&[]);
        let budget = extract.budget(&args, extract.validate(&args).unwrap());
        for schema in [r#"{"type":"boolean","type":"string"}"#,
            r#"{"$ref":"remote"}"#, r#"{"type":"string","pattern":".*"}"#] {
            assert!(extract.check_schema(schema.to_owned(), "document", budget).is_err());
        }
        let (grounded, _) = arguments(&["--source-membership"]);
        assert!(grounded.check_schema(r#"{"type":"boolean"}"#.to_owned(), "document", budget).is_err());
    }
    #[test]
    fn process_floor_includes_schema_source_vocabulary_and_full_result_storage() {
        let (extract, mut args) = arguments(&[]);
        let limits = extract.validate(&args).unwrap();
        args.memory_mib = limits.minimum_memory_bytes.saturating_sub(1) / MIB;
        assert!(extract.validate(&args).is_err());
        args.memory_mib = limits.minimum_memory_bytes.div_ceil(MIB);
        assert!(extract.validate(&args).is_ok());
    }
    #[cfg(not(feature = "asupersync-runtime"))]
    #[test]
    fn missing_runtime_refuses_without_reading_schema_source_or_model() {
        let mut output = Vec::new();
        assert_eq!(command(&[]).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
