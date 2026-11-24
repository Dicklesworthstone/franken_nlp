//! Fixed-schema extraction and complete correlated field review over NDJSON.
use super::*;
use crate::{
    batch::{BatchLimits, BatchSummary, BatchWork, extract::ExtractionMaskBudget,
        extract::semantic::int8::Int8SemanticExtractionBatchLimits},
    candidate_cli::{batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES}, stream::Failure},
    native_engine::{portable_int8::ProjectionWork, strict_int8::Int8Work},
    tokenizer::untrusted::SafeBpeWorkLimits,
};

pub(crate) struct BatchCommand {
    pub(in crate::candidate_cli) semantic: SemanticCommand,
    stream: StreamArgs,
    encoding: InvocationEncodingArgs,
}

#[derive(Args)]
struct StreamArgs {
    /// Nonempty records, including malformed input and flush controls.
    #[arg(long, default_value_t = 1000)] max_requests: u64,
    /// Complete input stream, including framing and rejected records.
    #[arg(long, default_value_t = 67_108_864)] max_total_input_bytes: u64,
    /// Complete output stream, including candidate provenance and final frames.
    #[arg(long, default_value_t = 67_108_864)] max_total_output_bytes: u64,
    #[arg(long, default_value_t = 1_048_576)] max_line_bytes: usize,
    /// Whole-stream forwards; defaults to ONE semantic-options document allowance.
    #[arg(long)] max_total_forward_positions: Option<u64>,
    /// Whole-stream projected logits; defaults to ONE document allowance.
    #[arg(long)] max_total_projected_logits: Option<u64>,
    /// Whole-stream attention pairs; defaults to ONE document allowance.
    #[arg(long)] max_total_attention_pairs: Option<u64>,
    /// Whole-stream projection dot products; defaults to ONE document allowance.
    #[arg(long)] max_total_dot_products: Option<u64>,
    /// Whole-stream multiply-accumulates; defaults to ONE document allowance.
    #[arg(long)] max_total_multiply_accumulates: Option<u64>,
    /// Whole-stream extraction masks; defaults to ONE --max-mask-node-visits allowance.
    #[arg(long)] max_total_mask_node_visits: Option<u64>,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    InvocationEncodingArgs::augment_args(StreamArgs::augment_args(super::definition()
        .name("extract-semantic-batch")
        .about("Extract and review ordered documents using one resident INT8 model")
        .long_about(concat!(
            "Process bounded ordered {id,text} NDJSON and exact {\"flush\":true} controls with one fixed schema, review policy, extraction compiler, judge, vocabulary and native engine. ",
            "Non-null task_args is refused before document encoding. Each successful record preserves the original independently validated JSON and every configured scalar claim's entailed/contradicted/unsupported/not_checked diagnostics. ",
            "A failed extraction or review publishes no combined document result. Required verbatim membership is checked against each actual original source. No fields are repaired or accepted by a second read. ",
            "One complete attempt's five native ceilings come from semantic-options.max_model_work. The six --max-total-* native/mask flags cap the whole stream and default to ONE document's allowance. ",
            "Every admitted attempt spends its complete conservative allowance before the first forward; failures, early EOS, flushes and --max-requests never refund or multiply it. ",
            "--compute-threads and --prefill-rows compose on one persistent numerical team across every extraction and review. Terminal completion follows physical join under the original invocation control. ",
            "Earlier complete records remain valid after a later failure; require both run_complete and exit 0 for whole-run success. This remains experimental, uncalibrated same-model diagnostic output.")))
        .mut_arg("input", |arg| arg.help("Ordered NDJSON file or '-' for stdin; one exact UTF-8 source per record"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded source bytes per original document"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Per-attempt extraction-mask cap; a separate --max-total-mask-node-visits caps the stream"))
        .mut_arg("max_checkpoints", |arg| arg.help("One cooperative checkpoint allowance for the complete stream"))
        .after_long_help(concat!(
            "Compact safe-bpe-v1 has two immutable envelopes: the existing --max-tokenizer-* policy spans original schema/source and all reviews in ONE complete attempt; ",
            "--max-total-tokenizer-* additionally bounds actual cumulative work across the invocation, including failed/cancelled preparation, dynamic reviews and reconstruction. ",
            "No tokenizer allowance resets at native handoff or flush. Byte-fallback remains the default and rejects tokenizer-only flags. ",
            "The bounded options file is read under caller preparation first; complete source/review/transport/native/output/team memory and all run ceilings are then admitted before schema, population or model IO. ",
            "Exact schema numbers, source bytes and offsets remain unchanged. No selected-row projection, cohorts, thinking, sampling, tools, automatic retry or semantic-accuracy promotion. Blocking IO remains cooperative.")))
}

#[derive(Clone, Copy)]
pub(in crate::candidate_cli) struct BatchEnvelope {
    pub native: Int8SemanticExtractionBatchLimits,
    pub transport: BatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
    pub encoding_work_limits: Option<SafeBpeWorkLimits>,
}

impl BatchCommand {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { semantic: SemanticCommand::from_matches(matches)?,
            stream: StreamArgs::from_arg_matches(matches)?,
            encoding: InvocationEncodingArgs::from_arg_matches(matches)? })
    }

    pub(in crate::candidate_cli) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        let s = &self.stream;
        if !(1..=100_000).contains(&s.max_requests)
            || s.max_line_bytes < self.semantic.host.max_input_bytes
            || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || [s.max_total_forward_positions, s.max_total_projected_logits, s.max_total_attention_pairs,
                s.max_total_dot_products, s.max_total_multiply_accumulates, s.max_total_mask_node_visits]
                .contains(&Some(0)) {
            return Err(CandidateError::Arguments);
        }
        self.encoding_work_limits()?;
        self.semantic.validate()
    }

    pub(in crate::candidate_cli) fn encoding_work_limits(&self)
        -> Result<Option<SafeBpeWorkLimits>, CandidateError> {
        self.encoding.limits(&self.semantic.encoding_policy()?)
    }

    pub(in crate::candidate_cli) fn policy_limits(&self, args: &CandidateArgs, limits: Limits,
        options: &Int8SemanticOptions) -> Result<BatchEnvelope, CandidateError> {
        let line_bytes = self.semantic.admit_options(args, limits, options)?;
        let s = &self.stream;
        let item = options.max_model_work;
        let work = Int8Work {
            forward_positions: s.max_total_forward_positions.unwrap_or(item.forward_positions),
            projected_logits: s.max_total_projected_logits.unwrap_or(item.projected_logits),
            attention_pairs: s.max_total_attention_pairs.unwrap_or(item.attention_pairs),
            projections: ProjectionWork {
                dot_products: s.max_total_dot_products.unwrap_or(item.projections.dot_products),
                multiply_accumulates: s.max_total_multiply_accumulates.unwrap_or(item.projections.multiply_accumulates),
            },
        };
        let native = Int8SemanticExtractionBatchLimits { max_model_work: work,
            masks: ExtractionMaskBudget { per_mask: self.semantic.host.masks(),
                max_visits_per_item: self.semantic.host.max_mask_node_visits,
                max_visits_per_run: s.max_total_mask_node_visits.unwrap_or(self.semantic.host.max_mask_node_visits) } };
        native.validate_item(item).map_err(|_| CandidateError::Arguments)?;
        let framing = s.max_requests.checked_add(4).and_then(|n| n.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let transport = BatchLimits { max_line_bytes: s.max_line_bytes,
            max_document_bytes: self.semantic.host.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: line_bytes,
            max_output_bytes: s.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits } };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = (line_bytes as u64).checked_add(FRAME_ALLOWANCE)
            .and_then(|n| n.checked_add(2 * IO_BUFFER_BYTES as u64)).ok_or(CandidateError::Arguments)?;
        Ok(BatchEnvelope { native, transport, output_bytes: s.max_total_output_bytes, io_bytes,
            encoding_work_limits: self.encoding_work_limits()? })
    }

    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { protocol: PROTOCOL, schema_version: 1, event: "error",
                    code: failure.code, exit_code: failure.exit, cancellation: failure.cancellation,
                    earlier_complete_records_may_exist: true };
                if publish(&report, FRAME_ALLOWANCE as usize, diagnostics).is_err() {
                    return ErrorCode::Generic.as_process_exit();
                }
                failure.exit.as_process_exit()
            }
        }
    }

    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W) -> Result<(), Failure> {
        let (args, limits) = self.validate().map_err(population_failure)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::semantic::corpus::execute(self, args, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(Failure::usage("semantic_batch_profile_unavailable")) }
    }
}

pub(in crate::candidate_cli) const PROTOCOL: &str = "fnlp-candidate-int8-semantic-extraction-batch-v1";

pub(in crate::candidate_cli) fn completed(summary: BatchSummary) -> Result<(), Failure> {
    if summary.failed == 0 { Ok(()) } else { Err(Failure::output("semantic_batch_items_failed")) }
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

#[cfg(test)] mod tests;
