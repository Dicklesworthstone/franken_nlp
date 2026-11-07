//! Live complete-source QA with an exact private question on every record.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork,
        source::grouped_int4::question::{Int4QuestionDocumentBatchConfig,
            memory::{Int4QuestionDocumentMemoryLimits, Int4QuestionDocumentMemoryRequirement}}},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
    tasks::answer::AnswerOptions,
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
pub(in crate::candidate_cli::int4) mod job;
#[cfg(test)]
mod tests;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct StreamArgs {
    /// Nonempty records, including malformed records and exact flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Every input byte, including oversized records, blank lines and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All emitted bytes, including public provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// Encoded NDJSON before LF, including source, question, JSON escaping and CR.
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
    /// Exact decoded question bytes per record; prompt/context fit is separate.
    #[arg(long, default_value_t = question::QUESTION_BYTES)]
    max_question_bytes: usize,
    /// Whole-run mask allowance across every admitted complete-document attempt.
    #[arg(long, default_value_t = 128_000_000_000_000)]
    max_run_mask_node_visits: u64,
    /// Independent whole-run source-map verification pool, including failed attempts.
    #[arg(long, default_value_t = 4_096_000)]
    max_run_source_fields: usize,
    #[arg(long, default_value_t = 16_384_000)]
    max_run_source_evidence_spans: usize,
    #[arg(long, default_value_t = 67_108_864_000)]
    max_run_source_scan_steps: u64,
    /// Whole-run QA citation pool spanning discovery and optional answer synthesis.
    #[arg(long, default_value_t = 4_096_000)]
    max_run_qa_citations: usize,
    #[arg(long, default_value_t = 16_384_000)]
    max_run_qa_evidence_spans: usize,
    #[arg(long, default_value_t = 67_108_864_000)]
    max_run_qa_scan_steps: u64,
    /// Run up to N independent discovery chunks together; 1..=64, no silent reduction.
    #[arg(long, value_name = "N")]
    cohort_chunks: Option<usize>,
    /// Execute up to N independent hierarchy groups together after a complete
    /// level preflight; independent of the discovery chunk schedule.
    #[arg(long, value_name = "N", requires = "hierarchical_answer")]
    cohort_hierarchy_groups: Option<usize>,
}
impl StreamArgs {
    fn map_verification(self) -> GroundingBudget {
        GroundingBudget { max_fields: self.max_run_source_fields,
            max_matches: self.max_run_source_evidence_spans, max_scan_steps: self.max_run_source_scan_steps }
    }
    fn verification(self) -> GroundingBudget {
        GroundingBudget { max_fields: self.max_run_qa_citations,
            max_matches: self.max_run_qa_evidence_spans, max_scan_steps: self.max_run_qa_scan_steps }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs {
    source: source::SourceArgs,
    mapping: MappingArgs,
    policy: question::QuestionPolicyArgs,
    synthesis: question::synthesis::SynthesisArgs,
    stream: StreamArgs,
}
impl std::fmt::Debug for CorpusArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("QuestionCorpusArgs") }
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let command = source::definition(Kind::Answer).name("answer-corpus")
        .about("Answer each record's own question across its complete original source")
        .long_about("Stream strict {id,text,task_args:{question}} NDJSON and exact {\"flush\":true} controls through one resident grouped-INT4 model. text is the COMPLETE original source; question is the exact private question for that record. Required task_args accepts only question, without options, passages, task or budget overrides. --options fixes complete AnswerOptions for the stream. Independently plan every nonblank source chunk against that record's actual question and exact source/template/schema capacity before its first native forward. Default output preserves every independent answer, abstention and whitespace range with original-coordinate citations. --synthesize-answer adds one native answer over ALL collected verified verbatim quotes; evidence must fit the complete passage, byte and context limits. --hierarchical-answer explicitly enables bounded, lossy quote-selection levels before the final answer when evidence exceeds one context. Every pass retains the record's original question; generated answer prose never becomes source evidence. All allowed hierarchy passes reserve work before discovery. Evidence, level, pass, tokenizer, verification, nonprogress or cancellation failures emit no partial document result. Documents execute in order; --cohort-chunks and --cohort-hierarchy-groups independently choose native discovery and hierarchy windows, each fully admitted and drained before the next phase. Neither widens per-task KV nor silently reduces the selected window. All five native work flags bound the ENTIRE RUN. Separate whole-run masks, map verification and QA verification debit before each attempt, including failed attempts; flush does not replenish them. Require run_complete AND exit 0 for whole-run success; earlier completed frames remain valid after later failure. Source conversion and independently pinned snapshots share this route. Optional snapshot saving follows fixed options/model/envelope admission and remains if later input fails. No question file/default, retrieval, semantic entailment guarantee, calibrated answerability, lossless hierarchy equivalence, retained job, selected-row projection, prompt prefill or automatic retry. Blocking IO observes cancellation cooperatively.")
        .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; every record supplies its own complete source and question"))
        .mut_arg("max_input_bytes", |arg| arg.default_value("1048576").help("Maximum decoded COMPLETE original source bytes per record; question has a separate ceiling"))
        .mut_arg("preparation_mib", |arg| arg.default_value("1024"))
        .mut_arg("options", |arg| arg.help("Complete AnswerOptions JSON file fixed across the stream; omitted uses shipped defaults"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Per-chunk grammar-mask allowance, including EOS"))
        .mut_arg("max_checkpoints", |arg| arg.help("Nonrenewable cooperative checkpoint allowance for the ENTIRE stream"));
    let command = StreamArgs::augment_args(question::QuestionPolicyArgs::augment_args(
        question::synthesis::SynthesisArgs::augment_args(MappingArgs::augment_args(command))));
    command
        .mut_arg("max_map_result_bytes", |arg| arg.help("Complete semantic answer document cap; transport and provenance framing are separate"))
        .mut_arg("max_total_mask_node_visits", |arg| arg.help("Complete-document mask allowance spanning discovery and optional synthesis"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream forward ceiling across every document and synthesis pass"))
        .mut_arg("max_projected_logits", |arg| arg.help("Whole-stream full-vocabulary projection ceiling"))
        .mut_arg("max_attention_pairs", |arg| arg.help("Whole-stream attention-pair ceiling"))
        .mut_arg("max_dot_products", |arg| arg.help("Whole-stream native grouped dot-product ceiling"))
        .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-stream multiply-accumulate ceiling"))
}

#[derive(Clone, Copy)]
pub(in crate::candidate_cli::int4) struct CorpusLimits {
    common: Int4Limits,
    config: Int4QuestionDocumentBatchConfig,
    transport: TransportLimits,
    memory: Int4QuestionDocumentMemoryRequirement,
    reduction_reserve_bytes: u64,
    output_bytes: u64,
    io_bytes: u64,
    cli_preparation_bytes: u64,
}
impl CorpusArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { source: source::SourceArgs::from_matches(matches)?, mapping: MappingArgs::from_arg_matches(matches)?,
            policy: question::QuestionPolicyArgs::from_arg_matches(matches)?,
            synthesis: question::synthesis::SynthesisArgs::from_arg_matches(matches)?,
            stream: StreamArgs::from_arg_matches(matches)? })
    }
    fn validate(&self, args: &Int4Args) -> Result<CorpusLimits, CandidateError> {
        // Numeric options do not change the declared memory dimensions. The
        // runtime replaces these shipped defaults with the actual bounded
        // options and repeats complete admission before model or population IO.
        self.validate_options(args, AnswerOptions::default())
    }
    fn validate_options(&self, args: &Int4Args, options: AnswerOptions) -> Result<CorpusLimits, CandidateError> {
        let mut common = self.source.validate(args)?;
        let s = self.stream;
        self.mapping.validate(args, &self.source, common, s.cohort_chunks)?;
        self.synthesis.validate()?;
        let verification = self.policy.verification()?;
        let mapping = self.mapping.limits(args, &self.source, s.cohort_chunks)?;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || s.max_line_bytes < args.max_input_bytes || s.max_line_bytes < s.max_question_bytes
            || !(1..=MAX_INPUT_BYTES).contains(&s.max_question_bytes)
            || !(1..=1_000_000_000_000_000_000).contains(&s.max_run_mask_node_visits)
            || s.max_run_mask_node_visits < mapping.max_mask_visits {
            return Err(CandidateError::Arguments);
        }
        verify_run_pool(mapping.verification, s.map_verification())?;
        verify_run_pool(verification, s.verification())?;
        let mut planning = self.source.planning(args);
        planning.max_input_bytes = planning.max_input_bytes.max(s.max_question_bytes);
        let planning = self.synthesis.planning(planning);
        let synthesis = self.synthesis.limits(planning)?;
        let hierarchy = self.synthesis.hierarchy()?;
        if let Some(max_groups) = s.cohort_hierarchy_groups {
            let limits = hierarchy.ok_or(CandidateError::Arguments)?;
            crate::tasks::source_planning::grouped_int4::long::cohort::hierarchy::Int4HierarchyCohortSchedule {
                max_groups, discovery_cohorts: s.cohort_chunks.is_some(),
            }.validate(limits.max_passes).map_err(|_| CandidateError::Arguments)?;
        }
        let budget = self.source.budget(args, common);
        let config = Int4QuestionDocumentBatchConfig { options, budget, planning, mapping, verification,
            synthesis, hierarchy, max_question_bytes: s.max_question_bytes, max_model_work: mapping.max_model_work,
            max_mask_visits: s.max_run_mask_node_visits, max_map_verification: s.map_verification(),
            max_verification: s.verification() };
        let per_chunk = sum(&[args.context_tokens as u64 * 32,
            u64::from(budget.max_grammar_states) * 512, shared::OPTIONS_BYTES as u64 * 2])?;
        let passes = self.mapping.max_chunks as u64 + config.continuation_passes() as u64;
        let preparation_floor = sum(&[times(per_chunk, passes)?, 256 * MIB,
            args.max_input_bytes as u64 * 64, (self.mapping.max_map_result_bytes as u64 + 4096) * 2])?;
        if common.preparation_bytes < preparation_floor { return Err(CandidateError::Arguments); }
        let framing = s.max_requests.checked_add(4).and_then(|n| n.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let transport = TransportLimits {
            max_line_bytes: s.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: self.mapping.max_map_result_bytes.checked_add(FRAME_ALLOWANCE as usize)
                .ok_or(CandidateError::Arguments)?,
            max_output_bytes: s.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: config.max_model_work.forward_positions,
                projected_logits: config.max_model_work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE, 2 * IO_BUFFER_BYTES as u64])?;
        let cli_preparation_bytes = sum(&[common.preparation_bytes, io_bytes])?;
        let reduction_reserve_bytes = mib(self.mapping.reduction_reserve_mib)?;
        let memory = Int4QuestionDocumentMemoryLimits { context_tokens: args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, preparation_reserve_bytes: common.preparation_bytes,
            io_reserve_bytes: io_bytes, reduction_reserve_bytes, cohort_chunks: s.cohort_chunks,
            hierarchy_groups: s.cohort_hierarchy_groups }
            .requirements(args.group_width, config, transport).map_err(|_| CandidateError::Arguments)?;
        let execution = sum(&[memory.native.kv_bytes, memory.workspace_bytes, memory.input_bytes,
            memory.document_workspace_bytes, memory.output_reservation_bytes])?;
        common.minimum_memory_bytes = sum(&[grouped_memory_floor(args, common, execution)?, io_bytes])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(CorpusLimits { common, config, transport, memory, reduction_reserve_bytes,
            output_bytes: s.max_total_output_bytes, io_bytes, cli_preparation_bytes })
    }
    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(diagnostics, "fnlp candidate answer-corpus: {}", diagnostic(error));
                error.exit_code()
            }
        }
    }
}
fn verify_run_pool(item: GroundingBudget, run: GroundingBudget) -> Result<(), CandidateError> {
    if !(1..=1_000_000_000).contains(&run.max_fields)
        || !(1..=1_000_000_000).contains(&run.max_matches)
        || !(1..=1_000_000_000_000_000_000).contains(&run.max_scan_steps)
        || item.max_fields > run.max_fields || item.max_matches > run.max_matches || item.max_scan_steps > run.max_scan_steps {
        return Err(CandidateError::Arguments);
    }
    Ok(())
}
fn diagnostic(error: CandidateError) -> &'static str {
    match error {
        CandidateError::Timeout => "cooperative corpus deadline expired; earlier completed frames may exist",
        CandidateError::Execution => "native corpus execution failed or was cancelled; earlier completed frames may exist",
        _ => error.message(),
    }
}
