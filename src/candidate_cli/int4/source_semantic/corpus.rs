//! Ordered cited summaries and passage answers with complete correlated review.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchSummary, BatchWork,
        extract::ExtractionMaskBudget,
        source::semantic::{Int4SemanticSourceBatchLimits, Int4SemanticSourceBatchTask}},
    candidate_cli::{batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES}, stream::Failure},
    grammar::runtime::JsonValidationSchema,
    native_engine::{portable_int8::ProjectionWork, strict_int4::Int4Work},
    tasks::{answer::AnswerOptions, ir::TaskBudget, source_planning::SourcePlanningLimits,
        summarize::SummaryOptions},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

const PROTOCOL: &str = "fnlp-candidate-int4-semantic-source-batch-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct BatchArgs {
    kind: Kind,
    semantic: SemanticSourceArgs,
    stream: StreamArgs,
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct StreamArgs {
    /// Whole stream's nonempty records, including malformed rows and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All emitted bytes, including candidate provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
    /// Whole-stream forward cap; defaults to ONE record's semantic-options ceiling.
    #[arg(long)]
    max_total_forward_positions: Option<u64>,
    /// Whole-stream projected-logit cap; defaults to ONE record's semantic-options ceiling.
    #[arg(long)]
    max_total_projected_logits: Option<u64>,
    /// Whole-stream attention-pair cap; defaults to ONE record's semantic-options ceiling.
    #[arg(long)]
    max_total_attention_pairs: Option<u64>,
    /// Whole-stream grouped-dot cap; defaults to ONE record's semantic-options ceiling.
    #[arg(long)]
    max_total_dot_products: Option<u64>,
    /// Whole-stream multiply-accumulate cap; defaults to ONE record's semantic-options ceiling.
    #[arg(long)]
    max_total_multiply_accumulates: Option<u64>,
    /// Whole-stream mask cap; defaults to --max-mask-node-visits for ONE record.
    #[arg(long)]
    max_total_mask_node_visits: Option<u64>,
}

pub(in crate::candidate_cli::int4) fn definitions() -> Vec<clap::Command> {
    [("summarize-semantic-batch", "Ordered {id,text} NDJSON; non-null task_args is refused"),
        ("answer-semantic-batch", "Ordered {id,text:question,task_args:{passages:[{id,text}]}} NDJSON")]
        .into_iter().zip(super::definitions()).map(|((name, input_help), command)| {
            StreamArgs::augment_args(command.name(name)
                .about("Generate and review ordered source records with one fixed policy and resident INT4 model")
                .long_about(concat!(
                    "Run ordered bounded NDJSON through one resident model, source compiler, judge, vocabulary and native workspace. Exact {\"flush\":true} controls drain an ID epoch. ",
                    "Summaries accept {id,text} and refuse every non-null task_args. Answers use text as the question and require task_args:{passages:[{id,text}]}; passages are data only. Task, options, budgets, review policy, token IDs and execution authority cannot be supplied per record. ",
                    "--options or the typed task default fixes generation settings for the entire stream. Required --semantic-options explicitly selects one complete Int4SourceSemanticOptions review policy. Every result preserves the original generation, tokens, citations and semantic_support=not_assessed plus the complete correlated review. Review sees the whole original source or every original passage, not just displayed citations. Question and passage IDs are not evidence. ",
                    "All dynamic statements are prepared and preflighted before any review forward. Failed generation, review, cancellation or delivery never becomes successful abstention or a partial combined result. Empty summaries and model-declared answer abstentions remain explicitly not_checked. ",
                    "semantic-options.max_model_work caps one complete record's five native axes; --max-mask-node-visits caps its generation masks. Six --max-total-* flags independently cap the whole stream, defaulting to ONE record's allowance. More requests grant no extra work; records, statements, flushes and failures never renew budgets. Admitted attempts retain their complete conservative debit. ",
                    "--max-input-bytes bounds the complete source input; for answers it includes question, joined passage text and canonical passage manifest. --max-line-bytes independently bounds the full NDJSON envelope. Every source and review must fit the declared context; no chunking, truncation or retrieval is performed. ",
                    "--compute-threads 2..64 retains one numerical team across all records and reviews. --input-encoding safe-bpe-v1 gives each record one tokenizer allowance across source/question/passages and all review prompts. Full transport, plan, native, combined-output and team memory is admitted before source-options/population/model IO; only the bounded fixed semantic-options file is read first. Source conversion, digest-pinned snapshots and optional snapshot saving share this route. ",
                    "Earlier complete records remain valid after a later error. Require run_complete AND exit 0 for whole-run success. Same-model review is an experimental uncalibrated diagnostic, never independent proof, automatic acceptance or field repair. No sampling, selected rows, prefill, cohorts, retry, durable resume, thinking or tools. Blocking IO observes cancellation cooperatively."))
                .mut_arg("input", |arg| arg.help(input_help))
                .mut_arg("options", |arg| arg.help("Fixed complete source-task options for every record; omitted uses the typed task default"))
                .mut_arg("max_input_bytes", |arg| arg.help("Complete decoded source-input cap per record, including QA question, joined passages and canonical manifest"))
                .mut_arg("max_checkpoints", |arg| arg.help("One cooperative checkpoint allowance for the complete stream"))
                .mut_arg("max_mask_node_visits", |arg| arg.help("Per-record generation-mask cap; --max-total-mask-node-visits separately caps the whole stream")))
        }).collect()
}

#[derive(Clone, Copy, Debug)]
struct BatchLimits {
    common: Int4Limits,
    task: TaskBudget,
    native: Int4SemanticSourceBatchLimits,
    transport: TransportLimits,
    output_bytes: u64,
    io_bytes: u64,
}
impl BatchArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(kind: Kind, matches: &clap::ArgMatches)
        -> Result<Self, clap::Error> {
        if !matches!(kind, Kind::Summarize | Kind::Answer) {
            return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "semantic source batch task refused"));
        }
        Ok(Self { kind, semantic: SemanticSourceArgs::from_matches(matches)?,
            stream: StreamArgs::from_arg_matches(matches)? })
    }
    fn validate(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        let s = self.stream;
        if !matches!(self.kind, Kind::Summarize | Kind::Answer)
            || !(1..=100_000).contains(&s.max_requests)
            || s.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || [s.max_total_forward_positions, s.max_total_projected_logits, s.max_total_attention_pairs,
                s.max_total_dot_products, s.max_total_multiply_accumulates, s.max_total_mask_node_visits]
                .contains(&Some(0)) {
            return Err(CandidateError::Arguments);
        }
        self.semantic.validate(args)
    }
    fn policy_limits(&self, args: &Int4Args, initial: Int4Limits, options: &Int4SourceSemanticOptions)
        -> Result<BatchLimits, CandidateError> {
        let common = self.semantic.admit_options(args, initial, options)?;
        let s = self.stream; let item = options.max_model_work;
        let work = Int4Work {
            forward_positions: s.max_total_forward_positions.unwrap_or(item.forward_positions),
            projected_logits: s.max_total_projected_logits.unwrap_or(item.projected_logits),
            attention_pairs: s.max_total_attention_pairs.unwrap_or(item.attention_pairs),
            projections: ProjectionWork {
                dot_products: s.max_total_dot_products.unwrap_or(item.projections.dot_products),
                multiply_accumulates: s.max_total_multiply_accumulates.unwrap_or(item.projections.multiply_accumulates),
            },
        };
        let native = Int4SemanticSourceBatchLimits { max_model_work: work,
            masks: ExtractionMaskBudget { per_mask: self.semantic.source.masks(),
                max_visits_per_item: self.semantic.source.mask_visits(),
                max_visits_per_run: s.max_total_mask_node_visits.unwrap_or(self.semantic.source.mask_visits()) } };
        native.validate_item(item).map_err(|_| CandidateError::Arguments)?;
        let framing = s.max_requests.checked_add(4).and_then(|n| n.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let transport = TransportLimits {
            max_line_bytes: s.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: common.result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: s.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE, 2 * IO_BUFFER_BYTES as u64])?;
        Ok(BatchLimits { task: self.semantic.source.budget(args, common), common, native, transport,
            output_bytes: s.max_total_output_bytes, io_bytes })
    }
    fn load_task(&self, json: Option<&str>, budget: TaskBudget, planning: SourcePlanningLimits,
        semantic: &Int4SourceSemanticOptions) -> Result<Int4SemanticSourceBatchTask, CandidateError> {
        let (task, schema) = match self.kind {
            Kind::Summarize => {
                let options: SummaryOptions = crate::candidate_cli::source::options(json)?;
                let schema = options.schema_source().map_err(|_| CandidateError::Planning)?;
                (Int4SemanticSourceBatchTask::Summarize { options, budget }, schema)
            }
            Kind::Answer => {
                let options: AnswerOptions = crate::candidate_cli::source::options(json)?;
                let schema = options.schema_source().map_err(|_| CandidateError::Planning)?;
                (Int4SemanticSourceBatchTask::Answer { options, budget }, schema)
            }
            _ => return Err(CandidateError::Arguments),
        };
        task.validate_semantic(semantic).map_err(|_| CandidateError::Planning)?;
        JsonValidationSchema::compile(&schema, planning.compiler).map_err(|_| CandidateError::Planning)?;
        Ok(task)
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        let kind = self.kind;
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => report_failure(kind, failure, diagnostics),
        }
    }
    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), Failure> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(Failure::usage("semantic_source_batch_profile_unavailable")) }
    }
}
fn completed(summary: BatchSummary) -> Result<(), Failure> {
    if summary.failed == 0 { Ok(()) }
    else { Err(Failure::output("semantic_source_batch_items_failed")) }
}
fn task_name(kind: Kind) -> &'static str {
    match kind { Kind::Summarize => "summarize-semantic", Kind::Answer => "answer-semantic",
        _ => "unsupported-semantic-source" }
}

#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, task: &'static str, event: &'static str,
    code: &'static str, exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    earlier_complete_records_may_exist: bool,
}
fn report_failure(kind: Kind, failure: Failure, diagnostics: &mut impl Write) -> ExitCode {
    let report = ErrorReport { protocol: PROTOCOL, schema_version: 1, task: task_name(kind), event: "error",
        code: failure.code, exit_code: failure.exit, cancellation: failure.cancellation,
        earlier_complete_records_may_exist: true };
    if publish(&report, 4096, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
    failure.exit.as_process_exit()
}
