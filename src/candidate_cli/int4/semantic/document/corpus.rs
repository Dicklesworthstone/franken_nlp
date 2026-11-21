//! Resident ordered populations of complete-document extraction and review.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchSummary, BatchWork,
        extract::semantic::document::Int4SemanticObjectDocumentBatchLimits},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
    native_engine::{portable_int8::ProjectionWork, strict_int4::Int4Work},
    validation::grounded_fields::GroundingBudget,
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs {
    document: SemanticObjectArgs,
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
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
    /// Whole-run native ceiling; defaults to one complete document's options ceiling.
    #[arg(long)]
    max_run_forward_positions: Option<u64>,
    #[arg(long)]
    max_run_projected_logits: Option<u64>,
    #[arg(long)]
    max_run_attention_pairs: Option<u64>,
    #[arg(long)]
    max_run_dot_products: Option<u64>,
    #[arg(long)]
    max_run_multiply_accumulates: Option<u64>,
    /// Whole-run masks; defaults to one complete document's declared mask cap.
    #[arg(long)]
    max_run_mask_node_visits: Option<u64>,
    #[arg(long)]
    max_run_extraction_fields: Option<usize>,
    #[arg(long)]
    max_run_extraction_evidence_spans: Option<usize>,
    #[arg(long)]
    max_run_extraction_scan_steps: Option<u64>,
}
pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    StreamArgs::augment_args(super::definition().name("extract-semantic-document-corpus")
        .about("Reconcile and review complete documents with one resident INT4 model")
        .long_about(concat!(
            "Stream bounded ordered {id,text} NDJSON and exact {\"flush\":true} controls using one resident model, original schema, object compiler, judge and vocabulary. ",
            "Each text is the COMPLETE original source. Non-null task_args is refused before document tokenization. ",
            "For every document, extract supported atomic root properties across all lossless chunks, reconcile exact complementary values against the original schema, then review every configured non-verbatim scalar claim against the COMPLETE original source. ",
            "The original object, chunk results and source evidence remain unchanged beside same-model-correlated, uncalibrated diagnostics. A failed extraction, reconciliation or review publishes no combined document result. ",
            "semantic-options.max_model_work bounds one complete extraction-plus-review attempt. The five --max-run-* native flags bound the entire stream independently and default to ONE document's allowance. ",
            "--max-total-mask-node-visits bounds one document; --max-run-mask-node-visits and --max-run-extraction-* bound masks and independent source verification across the entire stream. ",
            "Admitted failures retain the full conservative document debit. No allowance renews at a field, chunk, record or flush. ",
            "--compute-threads N keeps one persistent numerical team through every document and field review. Require run_complete AND exit 0 for whole-run success; the terminal footer follows physical worker join. Earlier completed records remain valid after later failure. ",
            "Compact safe-bpe-v1 uses one document ledger across extraction and all review prompts. Complete compiler, review, transport, native, output and team memory is admitted before schema/population/model IO; only the bounded fixed options file is read first. ",
            "Source conversion and independently pinned snapshots use the same native route. Optional snapshot saving follows fixed schema/policy admission. ",
            "No repair, automatic acceptance, calibrated truth guarantee, selected-row projection, chunk cohorts, packed prefill, sampling, thinking, tools, retained job or per-record policy override. Blocking IO observes cancellation cooperatively."))
        .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; text contains a COMPLETE original document"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded source bytes per complete original document"))
        .mut_arg("max_checkpoints", |arg| arg.help("One nonrenewable checkpoint allowance for the ENTIRE stream"))
        .mut_arg("compute_threads", |arg| arg.help("One persistent INT4 numerical team across the ENTIRE stream; terminal success follows physical join"))
        .after_long_help("The original map flags retain their per-document extraction scope. The semantic-options file supplies the combined extraction/review ceiling. Whole-run --max-run-* values are explicit and never multiply automatically with --max-requests."))
}

#[derive(Clone, Copy, Debug)]
struct CorpusLimits {
    document: extract_map::object::ObjectMapLimits,
    task: TaskBudget,
    review: Int4FaithfulnessMapLimits,
    work: Int4SemanticObjectDocumentBatchLimits,
    transport: TransportLimits,
    output_bytes: u64,
    io_bytes: u64,
}
impl CorpusArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { document: SemanticObjectArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }
    fn validate(&self, args: &Int4Args) -> Result<extract_map::object::ObjectMapLimits, CandidateError> {
        let s = self.stream;
        if !(1..=100_000).contains(&s.max_requests)
            || s.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || [s.max_run_forward_positions, s.max_run_projected_logits, s.max_run_attention_pairs,
                s.max_run_dot_products, s.max_run_multiply_accumulates, s.max_run_mask_node_visits,
                s.max_run_extraction_scan_steps].contains(&Some(0))
            || [s.max_run_extraction_fields, s.max_run_extraction_evidence_spans].contains(&Some(0)) {
            return Err(CandidateError::Arguments);
        }
        self.document.validate(args)
    }
    fn policy_limits(&self, args: &Int4Args, document: extract_map::object::ObjectMapLimits,
        options: &Int4SemanticOptions) -> Result<CorpusLimits, CandidateError> {
        options.validate().map_err(|_| CandidateError::Arguments)?;
        let common = document.map.common;
        let task = self.document.object.map.extraction.budget(args, common);
        let review = self.document.review_limits(args, common, options)?;
        let s = self.stream;
        let item = options.max_model_work;
        let mapping = document.mapping().mapping;
        let verification = mapping.verification;
        let work = Int4SemanticObjectDocumentBatchLimits {
            max_model_work: Int4Work {
                forward_positions: s.max_run_forward_positions.unwrap_or(item.forward_positions),
                projected_logits: s.max_run_projected_logits.unwrap_or(item.projected_logits),
                attention_pairs: s.max_run_attention_pairs.unwrap_or(item.attention_pairs),
                projections: ProjectionWork {
                    dot_products: s.max_run_dot_products.unwrap_or(item.projections.dot_products),
                    multiply_accumulates: s.max_run_multiply_accumulates.unwrap_or(item.projections.multiply_accumulates),
                },
            },
            max_mask_node_visits: s.max_run_mask_node_visits.unwrap_or(mapping.mapping.max_mask_visits),
            max_verification: GroundingBudget {
                max_fields: s.max_run_extraction_fields.unwrap_or(verification.max_fields),
                max_matches: s.max_run_extraction_evidence_spans.unwrap_or(verification.max_matches),
                max_scan_steps: s.max_run_extraction_scan_steps.unwrap_or(verification.max_scan_steps),
            },
        };
        work.validate_item(item, mapping.mapping.max_mask_visits, verification)
            .map_err(|_| CandidateError::Arguments)?;
        let framing = s.max_requests.checked_add(4).and_then(|count| count.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let output_line = options.limits.max_output_bytes.checked_add(FRAME_ALLOWANCE)
            .and_then(|bytes| usize::try_from(bytes).ok()).ok_or(CandidateError::Arguments)?;
        let transport = TransportLimits {
            max_line_bytes: s.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: output_line,
            max_output_bytes: s.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.max_model_work.forward_positions,
                projected_logits: work.max_model_work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE,
            2 * IO_BUFFER_BYTES as u64])?;
        Ok(CorpusLimits { document, task, review, work, transport, output_bytes: s.max_total_output_bytes, io_bytes })
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
        { let _ = (self, args, limits, input, output);
            Err(Failure::usage("semantic_object_corpus_profile_unavailable")) }
    }
}
fn completed(summary: BatchSummary) -> Result<(), Failure> {
    if summary.failed == 0 { Ok(()) } else { Err(Failure::output("semantic_object_corpus_items_failed")) }
}

#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, event: &'static str, code: &'static str,
    exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    earlier_complete_records_may_exist: bool,
}
fn report_failure(failure: Failure, diagnostics: &mut impl Write) -> ExitCode {
    let report = ErrorReport { protocol: "fnlp-candidate-int4-semantic-object-corpus-v1",
        schema_version: 1, event: "error", code: failure.code, exit_code: failure.exit,
        cancellation: failure.cancellation, earlier_complete_records_may_exist: true };
    if super::super::publish(&report, 4096, diagnostics).is_err() {
        return ErrorCode::Generic.as_process_exit();
    }
    failure.exit.as_process_exit()
}
