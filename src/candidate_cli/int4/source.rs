//! The four bounded source tasks share the explicit INT4 model loader.
//! Task options stay data; the code-owned source planner owns every prompt.
use super::*;
use crate::{
    candidate_cli::source::{self as shared, Kind},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
    native_engine::constrained_int4::sparse::Int4JsonSparseLimits,
    tasks::{BuiltInTask, ir::TaskBudget, source_planning::SourcePlanningLimits},
};

const TEXT_ONLY: &[&str] = &[
    "max_output_bytes", "seed", "temperature_milli", "top_k", "top_p_ppm", "logprobs",
    "prefill_rows", "min_new_tokens", "stop_suffixes", "banned_token_ids", "repetition_penalty_milli",
    "presence_penalty_milli", "frequency_penalty_milli", "logit_bias",
];

#[derive(Args, Clone, Eq, PartialEq)]
pub(super) struct SourceArgs {
    /// Complete task options in a bounded local JSON file. Unknown/partial options reject.
    #[arg(long, value_name = "FILE")]
    options: Option<PathBuf>,
    /// Complete typed task-result bytes, including source spans and work counters.
    #[arg(long, default_value_t = 1_048_576)]
    max_result_bytes: usize,
    #[arg(long, default_value_t = 4096)]
    max_grammar_states: u32,
    /// Aggregate grammar-mask traversal work, including terminal EOS.
    #[arg(long, default_value_t = 2_000_000_000)]
    max_mask_node_visits: u64,
    #[arg(long, default_value_t = 2_000_000)]
    mask_step_node_visits: u64,
    #[arg(skip)]
    max_selected_rows: Option<usize>,
}
impl std::fmt::Debug for SourceArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("SourceArgs") }
}
pub(super) fn definitions() -> Vec<clap::Command> {
    [Kind::Ner, Kind::Keyphrases, Kind::Summarize, Kind::Answer].into_iter().map(definition).collect()
}
/// Shared definitions retain full-head schedules; only these direct commands
/// can bind selected rows into a complete source task before host admission.
pub(super) fn one_off_definitions() -> Vec<clap::Command> {
    definitions().into_iter().map(|command| command
        .arg(clap::Arg::new("max_selected_rows").long("max-selected-rows")
            .value_name("N").value_parser(clap::value_parser!(usize))
            .help("Opt into all legal grammar-row projections; 1..=166144 rows per step including accepted EOS; excess refuses"))
        .after_long_help("Full-vocabulary projection remains the default. --max-selected-rows projects every legal token, never prunes the grammar, and refuses before projecting a set larger than the cap. The exact strategy, cap and source-task finalizer bind execution identity. Source spans, passage boundaries, citation verification, complete result limits and mask work remain unchanged. Summary prose and answer text can have wide legal sets, so a small cap may refuse them; no automatic fallback or retry occurs. --prefill-rows independently enables bounded layer-major prompt execution. This option checks only selected logits; actual-model parity and performance require separate evidence. Source batches expose their own fixed stream option, including document cohorts and packed prompt passes; maps, jobs and redaction retain full-vocabulary execution."))
        .collect()
}
pub(super) fn definition(kind: Kind) -> clap::Command {
    let about = match kind {
        Kind::Ner => "Find typed entities and all exact source occurrences using grouped INT4",
        Kind::Keyphrases => "Rank exact source keyphrases using grouped INT4",
        Kind::Summarize => "Summarize with byte-verified structural citations using grouped INT4",
        Kind::Answer => "Answer from explicit passages, or abstain, using grouped INT4",
    };
    let mut command = Int4Args::augment_args(clap::Command::new(kind.name()).about(about)
        .long_about("Run a bounded code-owned source task on an explicitly source-converted or digest-pinned snapshot INT4 model. NER preserves all compatible byte and Unicode-scalar spans; keyphrases remain exact source text. Summary and answer citations are independently checked against original source/passages: structural citation validity does not certify semantic support. Answer takes {\"question\":...,\"passages\":[{\"id\":...,\"text\":...}]} and may return an explicitly uncalibrated abstention. Failed decoding or validation is never converted to abstention. --options supplies the same complete typed options as the INT8 source task. Output is one completed result; all structured tasks use greedy constrained decoding with thinking disabled."))
        .mut_arg("input", |arg| arg.help("Exact UTF-8 source; answer instead requires a JSON question/passages object; '-' reads stdin"))
        .mut_arg("max_new_tokens", |arg| arg.default_value("512").help("Constrained output tokens INCLUDING terminal EOS"))
        .mut_arg("preparation_mib", |arg| arg.default_value("512"));
    for &id in TEXT_ONLY { command = command.mut_arg(id, |arg| arg.hide(true)); }
    SourceArgs::augment_args(command)
}
pub(super) fn task(kind: Kind) -> BuiltInTask {
    match kind { Kind::Ner => BuiltInTask::Ner, Kind::Keyphrases => BuiltInTask::Keyphrases,
        Kind::Summarize => BuiltInTask::Summarize, Kind::Answer => BuiltInTask::Answer }
}
impl SourceArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if matches.value_source("prefill_rows") == Some(clap::parser::ValueSource::CommandLine) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "this INT4 source workflow requires sequential prompt execution"));
        }
        let args = Self::from_matches_with_prefill(matches)?;
        if args.max_selected_rows.is_some() {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "this INT4 source workflow requires full-vocabulary head projection"));
        }
        Ok(args)
    }
    pub(super) fn from_matches_with_prefill(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if TEXT_ONLY.iter().filter(|&&id| id != "prefill_rows")
            .any(|id| matches.value_source(id) == Some(clap::parser::ValueSource::CommandLine)) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "INT4 source tasks accept constrained tokens and result limits, not generation controls"));
        }
        let mut args = Self::from_arg_matches(matches)?;
        args.max_selected_rows = matches.try_get_one::<usize>("max_selected_rows").ok().flatten().copied();
        Ok(args)
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        if args.prefill()?.is_some() { return Err(CandidateError::Arguments); }
        self.validate_full_projection_with_prefill(args)
    }
    /// Stream projection is sealed by its own planner. A one-off selected-row
    /// option cannot silently replace that policy when prefill is enabled.
    pub(super) fn validate_full_projection_with_prefill(&self, args: &Int4Args)
        -> Result<Int4Limits, CandidateError> {
        if self.max_selected_rows.is_some() { return Err(CandidateError::Arguments); }
        self.validate_with_prefill(args)
    }
    pub(super) fn validate_with_prefill(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        let _ = self.selected_rows()?;
        if self.options.as_ref().is_some_and(|path| path.as_os_str().is_empty() || path.as_os_str() == "-")
            || !(1..=MAX_CONTENT_BYTES).contains(&self.max_result_bytes)
            || !(1..=65_536).contains(&self.max_grammar_states)
            || args.max_new_tokens == 0 || args.max_new_tokens > 1024 || args.max_new_tokens >= args.context_tokens
            || !(1..=1_000_000_000).contains(&self.mask_step_node_visits)
            || !(1..=1_000_000_000_000).contains(&self.max_mask_node_visits)
            || args.max_output_bytes != 65_536
            || args.options(166_101)? != GenerationOptions::greedy(args.max_new_tokens, 65_536, 166_101) {
            return Err(CandidateError::Arguments);
        }
        let masks = self.mask_step_node_visits.checked_mul(args.max_new_tokens as u64)
            .ok_or(CandidateError::Arguments)?;
        if masks > self.max_mask_node_visits { return Err(CandidateError::Arguments); }
        // Bound vocabulary/source-index preparation, typed options and full
        // evidence copies independently of the native KV/output reservation.
        let extra = sum(&[128 * MIB, shared::OPTIONS_BYTES as u64 * 16,
            self.max_result_bytes as u64 * 16, u64::from(self.max_grammar_states) * 32])?;
        // The source prompt compiler reserves prompt + full output, including
        // EOS, against context. Keep its stricter bound in admission as well.
        args.validate_storage_with_prefill(self.max_result_bytes, args.max_new_tokens,
            args.context_tokens - args.max_new_tokens, 0, extra)
    }
    pub(super) fn budget(&self, args: &Int4Args, limits: Int4Limits) -> TaskBudget {
        TaskBudget { max_input_tokens: limits.max_prompt_tokens as u32,
            max_output_tokens: args.max_new_tokens as u32, max_output_bytes: self.max_result_bytes as u64,
            max_grammar_states: self.max_grammar_states, max_kv_bytes: limits.native.kv_bytes }
    }
    pub(super) fn planning(&self, args: &Int4Args) -> SourcePlanningLimits {
        SourcePlanningLimits { max_input_bytes: args.max_input_bytes, max_context_tokens: args.context_tokens,
            max_passages: shared::MAX_PASSAGES,
            compiler: CompileLimits { max_output_bytes: self.max_result_bytes,
                max_states: self.max_grammar_states as usize, ..CompileLimits::default() },
            source: SourceRuntimeLimits::default() }
    }
    pub(super) fn masks(&self) -> MaskWorkLimits {
        MaskWorkLimits { max_trie_node_visits: self.mask_step_node_visits as usize, checkpoint_interval_nodes: 256 }
    }
    pub(super) fn mask_visits(&self) -> u64 { self.max_mask_node_visits }
    pub(super) fn options_path(&self) -> Option<&std::path::Path> { self.options.as_deref() }
    fn selected_rows(&self) -> Result<Option<Int4JsonSparseLimits>, CandidateError> {
        self.max_selected_rows.map(|max_rows_per_step| {
            let limits = Int4JsonSparseLimits { max_rows_per_step };
            limits.validate().map_err(|_| CandidateError::Arguments)?;
            Ok(limits)
        }).transpose()
    }
    #[cfg(any(feature = "asupersync-runtime", test))]
    pub(super) fn configure_plan(&self, plan: crate::tasks::source_planning::grouped_int4::PreparedInt4SourceTask)
        -> Result<crate::tasks::source_planning::grouped_int4::PreparedInt4SourceTask, CandidateError> {
        if plan.selected_rows().is_some() { return Err(CandidateError::Planning); }
        match self.selected_rows()? {
            Some(limits) => plan.with_selected_rows(limits).map_err(|_| CandidateError::Planning),
            None => Ok(plan),
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
pub(super) fn planner() -> Result<(crate::tasks::source_planning::SourceTaskPlanner,
    crate::tasks::extract::ExtractionVocabulary), CandidateError> {
    let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let mut ends = controls.template_controls().entries().iter()
        .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
    let eos = ends.next().ok_or(CandidateError::Identity)?.id;
    if ends.next().is_some() { return Err(CandidateError::Identity); }
    let planner = crate::tasks::source_planning::SourceTaskPlanner::pinned(controls.template_controls(), eos)
        .map_err(|_| CandidateError::Planning)?;
    let vocabulary = crate::tasks::extract::ExtractionVocabulary::pinned(controls.template_controls())
        .map_err(|_| CandidateError::Planning)?;
    Ok((planner, vocabulary))
}

#[cfg(feature = "asupersync-runtime")]
pub(super) fn identity(model: &crate::native_engine::strict_int4::Int4ModelIdentity,
    planner: &crate::tasks::source_planning::SourceTaskPlanner, task: BuiltInTask)
    -> Result<crate::execution_identity::ExecutionIdentity, CandidateError> {
    use crate::execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode};
    use crate::native_engine::{generation::grouped_int4::bind_candidate_model, strict_int4::STRICT_INT4_EXECUTION};
    let absent = Sha256Digest::of_bytes(b"null");
    // Unbound model fields stay local and are replaced from the actual loaded
    // weights before this identity can reach planning or corpus admission.
    bind_candidate_model(ExecutionIdentity {
        schema_version: 1, source_revision: "unbound".to_owned(), logical_model_digest: absent,
        artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(), packing_set_digest: absent,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: task.spec().identity(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    }, model).map_err(|_| CandidateError::Identity)
}

#[cfg(test)]
mod tests;
