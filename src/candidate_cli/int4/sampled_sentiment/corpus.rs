//! Complete independent sentiment bundles over one owned resident stream.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchSummary, BatchWork,
        extract::ExtractionMaskBudget,
        sentiment::sampled::{SampledSentimentBatchLimits, sampled_sentiment_batch_preparation_bytes}},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

const MAX_REQUESTS: u64 = 100_000;
const MAX_TRANSPORT_BYTES: u64 = 16 * 1024 * MIB;
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct BatchArgs {
    sampled: SampledArgs,
    stream: StreamArgs,
}
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct StreamArgs {
    /// Whole-stream nonempty records, including malformed input and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Entire input transport, including blank lines, oversized records and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// Every output byte, including candidate provenance and terminal events.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// Maximum NDJSON bytes before LF, including JSON syntax and trailing CR.
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    StreamArgs::augment_args(super::definition().name("sentiment-sampled-batch")
        .about("Generate independent sentiment bundles from ordered NDJSON with one resident INT4 model")
        .long_about("Run bounded ordered {id,text,task_args?} NDJSON with exact {\"flush\":true} controls. task_args may contain only axes; omitted axes selects valence, arousal, dominance and approach. --seed and all sampler policy are fixed for the entire invocation. Each record retains every requested independent score/explanation sample and its descriptive aggregates; one failed sample rejects that complete record. The model, vocabulary and native buffers load once and remain resident. --compute-threads 2..64 uses one persistent CPU row team across every record, axis, sample and flush, joining every worker before run_complete. All five native work ceilings, --max-mask-node-visits, the checkpoint allowance and timeout cover the ENTIRE RUN and never renew at records or flushes. --input-encoding safe-bpe-v1 selects bounded lossless compact input under one tokenizer allowance per record; its encoded source is reused across the whole record. The same item ID and semantic request replay the same addressed samples under the same seed; distinct IDs have distinct private sample addresses. Input order, delivery sequence and flush epochs do not enter those addresses. IDs may be reused only after a drained flush. Every event carries non-authoritative candidate provenance. Require run_complete AND exit 0 for whole-run success; earlier complete records remain valid after later item failure. No partial-bundle success, calibrated confidence, quality claim, thinking, tools, cohort execution, layer-major prefill schedule or durable resume. Blocking IO is cooperative, not preemptible.")
        .mut_arg("input", |arg| arg.help("NDJSON file, or '-' for stdin; one bounded record at a time"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded UTF-8 document bytes per record"))
        .mut_arg("max_result_bytes", |arg| arg.help("Maximum complete sample bundle per record; candidate framing is additional"))
        .mut_arg("prefill_rows", |arg| arg.hide(true))
        .mut_arg("max_checkpoints", |arg| arg.help("Nonrenewable cooperative checkpoints for the ENTIRE stream"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream forward ceiling across all records, axes and samples"))
        .mut_arg("max_projected_logits", |arg| arg.help("Whole-stream full-vocabulary projection ceiling"))
        .mut_arg("max_attention_pairs", |arg| arg.help("Whole-stream attention-pair ceiling"))
        .mut_arg("max_dot_products", |arg| arg.help("Whole-stream actual grouped native dot-product ceiling"))
        .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-stream multiply-accumulate ceiling"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Whole-stream mask-node allowance across all records, samples and EOS")))
}

#[derive(Clone, Copy, Debug)]
struct BatchLimits {
    common: Int4Limits,
    transport: TransportLimits,
    native: SampledSentimentBatchLimits,
    output_bytes: u64,
    io_bytes: u64,
    preparation_reserve_bytes: u64,
}
impl BatchArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<Self, clap::Error> {
        if matches.value_source("prefill_rows") == Some(clap::parser::ValueSource::CommandLine) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "sampled sentiment batch does not support a layer-major prefill schedule"));
        }
        Ok(Self { sampled: SampledArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }

    fn validate(self, args: &Int4Args) -> Result<BatchLimits, CandidateError> {
        let stream = self.stream;
        if args.prefill()?.is_some()
            || !(1..=MAX_REQUESTS).contains(&stream.max_requests)
            || stream.max_line_bytes < args.max_input_bytes
            || !(1..=MAX_LINE_BYTES).contains(&stream.max_line_bytes)
            || !(1..=MAX_TRANSPORT_BYTES).contains(&stream.max_total_input_bytes)
            || !(1..=MAX_TRANSPORT_BYTES).contains(&stream.max_total_output_bytes) {
            return Err(CandidateError::Arguments);
        }
        // The one-shot quote already owns both caller and host preparation,
        // the complete bundle output, one sampler, native memory and one team.
        // Corpus staging is additional; neither requests nor samples multiply
        // model, KV, sampler or worker reservations.
        let mut common = self.sampled.validate(args)?;
        let budget = self.sampled.budget(args, common, SentimentAxis::ALL.len())?;
        let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens)?;
        let preparation = sampled_sentiment_batch_preparation_bytes(budget,
            self.sampled.task_limits(args), &encoding).map_err(|_| CandidateError::Arguments)?;
        if preparation > common.preparation_bytes { return Err(CandidateError::Arguments); }
        let work = self.sampled.work_ceiling();
        let native = SampledSentimentBatchLimits {
            max_item_work: work, max_model_work: work,
            masks: ExtractionMaskBudget { per_mask: self.sampled.masks(),
                max_visits_per_item: self.sampled.max_mask_node_visits,
                max_visits_per_run: self.sampled.max_mask_node_visits },
            max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        };
        native.validate().map_err(|_| CandidateError::Arguments)?;
        let framing = stream.max_requests.checked_add(4).and_then(|n| n.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let transport = TransportLimits {
            max_line_bytes: stream.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: stream.max_total_input_bytes, max_requests: stream.max_requests,
            max_output_line_bytes: common.result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: stream.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE,
            2 * IO_BUFFER_BYTES as u64])?;
        // Match CorpusLimits::reservation_bytes, excluding its preparation
        // owner, which is already included in the sampled one-shot floor.
        let corpus_staging = sum(&[times(transport.max_line_bytes as u64, 8)?,
            times(transport.max_output_line_bytes as u64, 4)?,
            times(transport.max_epoch_ids as u64, 128)?, transport.max_epoch_id_bytes as u64, io_bytes])?;
        common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, corpus_staging])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(BatchLimits { common, transport, native, output_bytes: stream.max_total_output_bytes,
            io_bytes, preparation_reserve_bytes: common.preparation_bytes })
    }

    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let _ = writeln!(diagnostics, "fnlp candidate sampled sentiment batch: {}", failure.code);
                failure.exit.as_process_exit()
            }
        }
    }

    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), Failure> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output);
            Err(Failure::usage("sampled_sentiment_batch_profile_unavailable")) }
    }
}

fn times(value: u64, count: u64) -> Result<u64, CandidateError> {
    value.checked_mul(count).ok_or(CandidateError::Arguments)
}
fn completed(summary: BatchSummary) -> Result<(), Failure> {
    if summary.failed == 0 { Ok(()) }
    else { Err(Failure::output("sampled_sentiment_batch_items_failed")) }
}
