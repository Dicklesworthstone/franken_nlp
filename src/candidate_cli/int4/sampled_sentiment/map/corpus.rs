//! Complete sampled documents over one bounded resident NDJSON invocation.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchSummary, BatchWork},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs {
    document: MapArgs,
    stream: StreamArgs,
}

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct StreamArgs {
    /// Nonempty records across the whole run, including errors and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Entire input transport, including blank lines, oversized records and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// Every output byte, including candidate provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// NDJSON bytes before LF, including JSON escaping and a trailing CR.
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    StreamArgs::augment_args(super::definition().name("sentiment-sampled-corpus")
        .about("Sample every chunk of ordered complete documents using one resident INT4 model")
        .long_about("Stream bounded ordered {id,text,task_args?} NDJSON and exact {\"flush\":true} controls. Each document is losslessly partitioned and every chunk's complete independent score/explanation bundle is preflighted before its first native forward. Results retain all original byte/scalar spans and exact scalar-weighted document replicates. Omitted task_args selects all four axes; task_args accepts axes and an optional complete task budget that may only tighten the invocation ceiling. Seed, sample count, sampler, chunk policy and model identity are fixed for the run. Distinct item IDs receive distinct private draws; reorder and drained flushes do not change the same semantic item's draws. Weights, vocabulary and native buffers remain resident across documents. --compute-threads 2..64 keeps one numerical team across the entire stream; alternatively --prefill-rows 1..64 uses serial layer-major prompts. --reuse-prefill retains only the current chunk/axis prefix across its independent samples. All five native work flags and --max-mask-node-visits cover the ENTIRE RUN, including failed admitted documents, and never renew at records, chunks, samples or flushes. --input-encoding safe-bpe-v1 gives each complete document one bounded tokenizer allowance shared by chunk probes and actual plans. Every successful record contains a complete document; earlier completed records remain valid after a later failure. Require run_complete AND exit 0 for whole-run success. Source conversion and independently pinned snapshots share the route. No snapshot saving, retained jobs, cross-document cohorts, partial document success, calibrated confidence, full-context equivalence, thinking or tools. Blocking IO observes cancellation cooperatively.")
        .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; one bounded complete document per record"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded complete UTF-8 source bytes per document"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete independent sample bundle per source chunk"))
        .mut_arg("max_map_result_bytes", |arg| arg.help("Complete document result; event and provenance framing are additional"))
        .mut_arg("max_checkpoints", |arg| arg.help("Nonrenewable cooperative checkpoints for the ENTIRE stream"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream cold forward ceiling across documents, chunks, axes and samples"))
        .mut_arg("max_projected_logits", |arg| arg.help("Whole-stream full-vocabulary projection ceiling"))
        .mut_arg("max_attention_pairs", |arg| arg.help("Whole-stream attention-pair ceiling"))
        .mut_arg("max_dot_products", |arg| arg.help("Whole-stream grouped dot-product ceiling"))
        .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-stream multiply-accumulate ceiling"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("One whole-stream mask allowance across all documents, chunks, samples and EOS")))
}

#[derive(Clone, Copy, Debug)]
struct CorpusLimits {
    common: Int4Limits,
    document: MapLimits,
    transport: TransportLimits,
    memory: SampledSentimentMapMemoryRequirement,
    cli_preparation_bytes: u64,
    output_bytes: u64,
    io_bytes: u64,
}

impl CorpusArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<Self, clap::Error> {
        Ok(Self { document: MapArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }

    fn validate(self, args: &Int4Args) -> Result<CorpusLimits, CandidateError> {
        let stream = self.stream;
        if !(1..=100_000).contains(&stream.max_requests)
            || stream.max_line_bytes < args.max_input_bytes
            || !(1..=4 * 1024 * 1024).contains(&stream.max_line_bytes)
            || !(1..=16 * 1024 * MIB).contains(&stream.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&stream.max_total_output_bytes) {
            return Err(CandidateError::Arguments);
        }
        let document = self.document.validate(args)?;
        let mut common = document.common;
        let work = document.mapping.max_model_work;
        let framing = stream.max_requests.checked_add(4)
            .and_then(|n| n.checked_mul(FRAME_ALLOWANCE)).ok_or(CandidateError::Arguments)?;
        let transport = TransportLimits {
            max_line_bytes: stream.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: stream.max_total_input_bytes, max_requests: stream.max_requests,
            max_output_line_bytes: document.envelope_bytes,
            max_output_bytes: stream.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE,
            2 * IO_BUFFER_BYTES as u64])?;
        // Same transport/IO terms as CorpusLimits::reservation_bytes, without
        // duplicating the shared preparation already owned by the map quote.
        let corpus_staging = sum(&[times(transport.max_line_bytes as u64, 8)?,
            times(transport.max_output_line_bytes as u64, 4)?,
            times(transport.max_epoch_ids as u64, 128)?, transport.max_epoch_id_bytes as u64, io_bytes])?;
        let mut memory = document.memory;
        memory.input_bytes = sum(&[memory.input_bytes, corpus_staging])?;
        let prefill = common.prefill.map(Int4PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        memory.workspace_bytes = sum(&[memory.workspace_bytes, prefill, common.compute.workspace_bytes])?;
        let cli_preparation_bytes = common.preparation_bytes;
        let resident = sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES])?;
        let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        let execution = sum(&[memory.input_bytes, memory.native.kv_bytes,
            memory.workspace_bytes, memory.output_reservation_bytes])?;
        common.minimum_memory_bytes = sum(&[cli_preparation_bytes, resident,
            loading.max(execution), CLEANUP_BYTES])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(CorpusLimits { common, document, transport, memory, cli_preparation_bytes,
            output_bytes: stream.max_total_output_bytes, io_bytes })
    }

    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let _ = writeln!(diagnostics, "fnlp candidate complete sampled sentiment corpus: {}", failure.code);
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
            Err(Failure::usage("sampled_sentiment_corpus_profile_unavailable")) }
    }
}

fn times(value: u64, count: u64) -> Result<u64, CandidateError> {
    value.checked_mul(count).ok_or(CandidateError::Arguments)
}
fn completed(summary: BatchSummary) -> Result<(), Failure> {
    if summary.failed == 0 { Ok(()) }
    else { Err(Failure::output("sampled_sentiment_corpus_items_failed")) }
}
