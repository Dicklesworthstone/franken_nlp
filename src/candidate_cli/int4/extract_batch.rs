//! Ordered schema/source extraction over one resident grouped-INT4 model.
//! Finite transport, native and mask budgets never renew at an epoch boundary.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork,
        extract::{ExtractionBatchArgs, ExtractionBatchGrounding, ExtractionMaskBudget,
            grouped_int4::Int4ExtractionBatchLimits}},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
    grammar::runtime::JsonProgram,
    native_engine::{constrained_int4, strict_int4::Int4Work},
    tasks::ir::TaskBudget,
    validation::{JsonLimits, JsonValue, parse_json_with_limits},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

const MAX_REQUESTS: u64 = 100_000;
const MAX_TRANSPORT_BYTES: u64 = 16 * 1024 * MIB;
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BatchArgs {
    pub(super) extraction: extract::ExtractionArgs,
    stream: StreamArgs,
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct StreamArgs {
    /// Whole-stream nonempty records, including malformed input and flush commands.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Whole input transport, including oversized records, blank lines and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All emitted bytes, including candidate provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// Maximum NDJSON bytes before LF, including JSON syntax and any trailing CR.
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
}

pub(super) fn definition() -> clap::Command {
    StreamArgs::augment_args(extract::definition().name("extract-batch")
        .about("Extract one shared schema from bounded NDJSON using one resident INT4 model")
        .long_about("Process ordered {id,text} NDJSON and exact {\"flush\":true} controls with one resident source-converted or digest-pinned snapshot INT4 model. --schema supplies one exact shared schema; --source-membership binds its annotated verbatim fields to EACH original document. Any non-null task_args is rejected before document tokenization, even if it repeats the shared schema or budgets. Input is read incrementally and each result/error is flushed before reading the next record. IDs are unique within each bounded epoch and reset only after a completed flush. Every event has candidate provenance. Require run_complete AND exit 0 for whole-run success; run_complete alone reports a drained protocol and can include failed documents. Earlier complete results remain valid after a later failure. Native work, mask visits, checkpoints, input and output ceilings never renew per document or epoch. Execution is serial item-local full-vocabulary greedy extraction, with no sampler, repair retry, layer-major prefill, durable resume or semantic-accuracy claim. Blocking IO is cooperative, not preemptible.")
        .mut_arg("input", |arg| arg.help("NDJSON file, or '-' for stdin; read one bounded record at a time"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded UTF-8 document bytes per record"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoint allowance for the ENTIRE corpus execution")))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BatchLimits {
    pub common: Int4Limits,
    pub transport: TransportLimits,
    pub native: Int4ExtractionBatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
    pub preparation_reserve_bytes: u64,
}
impl BatchArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { extraction: extract::ExtractionArgs::from_matches(matches)?,
            stream: StreamArgs::from_arg_matches(matches)? })
    }

    pub(super) fn validate(&self, args: &Int4Args) -> Result<BatchLimits, CandidateError> {
        let stream = self.stream;
        if !(1..=MAX_REQUESTS).contains(&stream.max_requests)
            || stream.max_line_bytes < args.max_input_bytes
            || !(1..=MAX_LINE_BYTES).contains(&stream.max_line_bytes)
            || !(1..=MAX_TRANSPORT_BYTES).contains(&stream.max_total_input_bytes)
            || !(1..=MAX_TRANSPORT_BYTES).contains(&stream.max_total_output_bytes) {
            return Err(CandidateError::Arguments);
        }
        // Retain source/snapshot exclusivity, real hardware detection, schema
        // path checks and the extraction-only decoder policy before any IO.
        let mut common = self.extraction.validate(args)?;
        let one = constrained_int4::planned_work(args.group_width, common.max_prompt_tokens,
            args.max_new_tokens).map_err(|_| CandidateError::Arguments)?;
        // Each item has its OWN attention triangle. Scaling the already
        // computed work must not square a fictitious concatenated corpus.
        let work = scale_work(one, stream.max_requests)?;
        let native = Int4ExtractionBatchLimits {
            max_model_work: work,
            masks: ExtractionMaskBudget { per_mask: self.extraction.masks(),
                max_visits_per_item: self.extraction.mask_visits(),
                max_visits_per_run: times(self.extraction.mask_visits(), stream.max_requests)? },
        };
        // One start, at most one event per request, EOF flush and terminal;
        // reserve one additional frame for failure at the next nonempty line.
        let frames = stream.max_requests.checked_add(4).ok_or(CandidateError::Arguments)?;
        let framing = times(frames, FRAME_ALLOWANCE)?;
        let transport = TransportLimits {
            max_line_bytes: stream.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: stream.max_total_input_bytes, max_requests: stream.max_requests,
            max_output_line_bytes: common.result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: stream.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions,
                projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        native.validate().map_err(|_| CandidateError::Arguments)?;
        // The CLI holds its preparation guard through hosted completion. The
        // host independently prices its compiler/source/vocabulary reserve,
        // NDJSON copies, ID set, canonical staging and owned IO buffers.
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE,
            IO_BUFFER_BYTES as u64])?;
        let corpus_reservation = sum(&[
            times(transport.max_line_bytes as u64, 8)?,
            times(transport.max_output_line_bytes as u64, 4)?,
            times(transport.max_epoch_ids as u64, 128)?,
            transport.max_epoch_id_bytes as u64, preparation_reserve_bytes, io_bytes,
        ])?;
        common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, corpus_reservation])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(BatchLimits { common, transport, native, output_bytes: stream.max_total_output_bytes,
            io_bytes, preparation_reserve_bytes })
    }

    /// Preserve exact schema bytes. Source membership is compiled against
    /// EACH real document later; an invented preflight source proves nothing.
    pub(super) fn load_defaults(&self, schema: String, ceiling: TaskBudget)
        -> Result<ExtractionBatchArgs, CandidateError> {
        ceiling.validate().map_err(|_| CandidateError::Planning)?;
        let request = crate::candidate_cli::extract::arguments(schema,
            self.extraction.source_membership(), ceiling)?;
        let declaration = parse_json_with_limits(&request.schema, JsonLimits {
            max_input_bytes: crate::candidate_cli::extract::SCHEMA_BYTES,
            max_string_lexeme_bytes: crate::candidate_cli::extract::SCHEMA_BYTES,
            ..JsonLimits::default()
        }).map_err(|_| CandidateError::Planning)?;
        if !matches!(declaration, JsonValue::Object(_)) { return Err(CandidateError::Planning); }
        if request.grounding == ExtractionBatchGrounding::Structural {
            JsonProgram::compile(&request.schema, self.extraction.compiler())
                .map_err(|_| CandidateError::Planning)?;
        }
        Ok(request)
    }

    pub(super) fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W,
    ) -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

fn times(value: u64, count: u64) -> Result<u64, CandidateError> {
    value.checked_mul(count).ok_or(CandidateError::Arguments)
}
fn scale_work(mut work: Int4Work, count: u64) -> Result<Int4Work, CandidateError> {
    if count == 0 { return Err(CandidateError::Arguments); }
    for value in [&mut work.forward_positions, &mut work.projected_logits, &mut work.attention_pairs,
        &mut work.projections.dot_products, &mut work.projections.multiply_accumulates] {
        *value = times(*value, count)?;
    }
    Ok(work)
}
