//! Ordered extraction plus explicit same-model field review on one resident model.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchSummary, BatchWork,
        extract::{ExtractionBatchArgs, ExtractionMaskBudget,
            semantic::Int4SemanticExtractionBatchLimits}},
    candidate_cli::{batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES}, stream::Failure},
    grammar::runtime::JsonValidationSchema,
    native_engine::{portable_int8::ProjectionWork, strict_int4::Int4Work},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct BatchArgs {
    semantic: SemanticArgs,
    stream: StreamArgs,
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct StreamArgs {
    /// Entire stream's nonempty records, including malformed input and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All emitted bytes, including candidate provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
    /// Whole-stream forward cap; defaults to ONE document's semantic-options ceiling.
    #[arg(long)]
    max_total_forward_positions: Option<u64>,
    /// Whole-stream projected-logit cap; defaults to ONE document's semantic-options ceiling.
    #[arg(long)]
    max_total_projected_logits: Option<u64>,
    /// Whole-stream attention-pair cap; defaults to ONE document's semantic-options ceiling.
    #[arg(long)]
    max_total_attention_pairs: Option<u64>,
    /// Whole-stream grouped-dot cap; defaults to ONE document's semantic-options ceiling.
    #[arg(long)]
    max_total_dot_products: Option<u64>,
    /// Whole-stream multiply-accumulate cap; defaults to ONE document's semantic-options ceiling.
    #[arg(long)]
    max_total_multiply_accumulates: Option<u64>,
    /// Whole-stream mask cap; defaults to --max-mask-node-visits for ONE document.
    #[arg(long)]
    max_total_mask_node_visits: Option<u64>,
}
pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    StreamArgs::augment_args(super::definition().name("extract-semantic-batch")
        .about("Extract and review ordered documents with one fixed schema and resident INT4 model")
        .long_about(concat!(
            "Process bounded ordered {id,text} NDJSON and exact {\"flush\":true} controls using one resident model, schema compiler, judge and vocabulary. ",
            "--schema and --semantic-options fix the exact extraction declaration and explicitly opted-in correlated field-review policy for the whole run. Non-null task_args is refused before document tokenization. ",
            "Each result preserves the original independently validated extraction unchanged together with every configured scalar claim's entailed/contradicted/unsupported/not_checked diagnostics. Null, verbatim, unmapped and empty values remain not_checked; absent optional properties are not invented. ",
            "All dynamic field judgments are prepared and preflighted before the first review forward. A failed extraction or review publishes no combined document result. --source-membership binds annotated verbatim values to each real source. ",
            "One complete document's five native ceilings come from semantic-options.max_model_work; --max-mask-node-visits caps that document's extraction masks. The six --max-total-* work flags independently cap the entire stream, defaulting to one document's allowance. They never multiply with --max-requests or renew at records, fields, failures or flushes. Failed admitted documents retain their conservative full-attempt debit. ",
            "--compute-threads 2..64 keeps one numerical team through every extraction and review, with terminal completion after physical join. Compact safe-bpe-v1 gives each document one tokenizer ledger shared by its exact schema/source and every review prompt. All compiler, transport, native, combined-output and team memory is admitted before schema/population/model IO; only the bounded fixed options file is read first. ",
            "Weights may be source-converted or loaded from an independently pinned snapshot. Optional snapshot saving follows fixed-schema and policy admission; individual source-bound plans are checked later. ",
            "Earlier complete records remain valid after a later error. Require run_complete AND exit 0 for whole-run success. This is an experimental, uncalibrated same-model diagnostic, not independent proof or automatic acceptance. No field repair, truncation, sampling, prefill, cohorts, selected-row projection, retry, durable resume, thinking or tools. Blocking IO observes cancellation cooperatively."))
        .mut_arg("input", |arg| arg.help("Ordered NDJSON file or '-' for stdin; one exact UTF-8 document per record"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded UTF-8 source bytes per document"))
        .mut_arg("max_checkpoints", |arg| arg.help("One cooperative checkpoint allowance for the complete stream"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Per-document extraction-mask cap; --max-total-mask-node-visits separately caps the whole stream")))
}

#[derive(Clone, Copy, Debug)]
struct BatchLimits {
    common: Int4Limits,
    task: TaskBudget,
    native: Int4SemanticExtractionBatchLimits,
    transport: TransportLimits,
    output_bytes: u64,
    io_bytes: u64,
}
impl BatchArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { semantic: SemanticArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }
    fn validate(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        let s = self.stream;
        if !(1..=100_000).contains(&s.max_requests)
            || s.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || [s.max_total_forward_positions, s.max_total_projected_logits, s.max_total_attention_pairs,
                s.max_total_dot_products, s.max_total_multiply_accumulates, s.max_total_mask_node_visits]
                .contains(&Some(0)) {
            return Err(CandidateError::Arguments);
        }
        self.semantic.validate_full_projection(args)
    }

    /// The fixed options supply the complete combined-result and per-document
    /// work ceilings. Runtime admission adds the exact hosted memory quote
    /// before opening schema, source population or model files.
    fn policy_limits(&self, args: &Int4Args, initial: Int4Limits, options: &Int4SemanticOptions)
        -> Result<BatchLimits, CandidateError> {
        let common = self.semantic.admit_options(args, initial, options)?;
        let s = self.stream;
        let item = options.max_model_work;
        let work = Int4Work {
            forward_positions: s.max_total_forward_positions.unwrap_or(item.forward_positions),
            projected_logits: s.max_total_projected_logits.unwrap_or(item.projected_logits),
            attention_pairs: s.max_total_attention_pairs.unwrap_or(item.attention_pairs),
            projections: ProjectionWork {
                dot_products: s.max_total_dot_products.unwrap_or(item.projections.dot_products),
                multiply_accumulates: s.max_total_multiply_accumulates.unwrap_or(item.projections.multiply_accumulates),
            },
        };
        let native = Int4SemanticExtractionBatchLimits {
            max_model_work: work,
            masks: ExtractionMaskBudget { per_mask: self.semantic.extraction.masks(),
                max_visits_per_item: self.semantic.extraction.mask_visits(),
                max_visits_per_run: s.max_total_mask_node_visits.unwrap_or(self.semantic.extraction.mask_visits()) },
        };
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
        Ok(BatchLimits { task: self.semantic.extraction.budget(args, common), common, native, transport,
            output_bytes: s.max_total_output_bytes, io_bytes })
    }

    fn load_defaults(&self, schema: String, task: TaskBudget, options: &Int4SemanticOptions)
        -> Result<ExtractionBatchArgs, CandidateError> {
        let defaults = self.semantic.extraction.shared_defaults(schema, task)?;
        // Declaration-only validation permits verbatim annotations without
        // inventing a source. Actual membership is checked per real document.
        let declaration = JsonValidationSchema::compile(&defaults.schema, self.semantic.extraction.compiler())
            .map_err(|_| CandidateError::Planning)?;
        if declaration.requires_verbatim_source()
            != (defaults.grounding == crate::batch::extract::ExtractionBatchGrounding::SourceMembership) {
            return Err(CandidateError::Planning);
        }
        options.validate_schema(declaration.root()).map_err(|_| CandidateError::Planning)?;
        Ok(defaults)
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => report_failure(failure, diagnostics),
        }
    }
    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), Failure> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(Failure::usage("semantic_batch_profile_unavailable")) }
    }
}
fn completed(summary: BatchSummary) -> Result<(), Failure> {
    if summary.failed == 0 { Ok(()) }
    else { Err(Failure::output("semantic_batch_items_failed")) }
}

#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str,
    schema_version: u32,
    event: &'static str,
    code: &'static str,
    exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    earlier_complete_records_may_exist: bool,
}
fn report_failure(failure: Failure, diagnostics: &mut impl Write) -> ExitCode {
    let report = ErrorReport { protocol: "fnlp-candidate-int4-semantic-extraction-batch-v1",
        schema_version: 1, event: "error", code: failure.code, exit_code: failure.exit,
        cancellation: failure.cancellation, earlier_complete_records_may_exist: true };
    if publish(&report, 4096, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
    failure.exit.as_process_exit()
}
