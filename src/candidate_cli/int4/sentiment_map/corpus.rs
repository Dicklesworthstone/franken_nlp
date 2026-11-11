//! Ordered complete-document sentiment with one resident native model.
//! Every native work axis covers the entire stream, including failed attempts.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork,
        sentiment::grouped_int4::SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES},
    candidate_cli::{batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
        scored_batch::{self as shared, Defaults, ScoreTask}},
    tasks::sentiment::{SentimentPolicy, batch::SentimentBatchArgs},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Args, Clone, Eq, PartialEq)]
struct StreamArgs {
    /// Fixed private {axes?,policy?} JSON; no source, budget or model identity.
    #[arg(long, value_name = "FILE")]
    defaults: Option<PathBuf>,
    /// Hard axis-count ceiling for defaults and every per-record replacement.
    #[arg(long, default_value_t = 4)]
    max_axes: usize,
    /// All nonempty records, including malformed records and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Entire transport, including oversized records, blank lines and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// Every emitted byte, including candidate provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// NDJSON bytes before LF, including JSON escaping and a trailing CR.
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
}

#[derive(Clone, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs {
    document: MapArgs,
    stream: StreamArgs,
}
impl std::fmt::Debug for CorpusArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SentimentCorpusArgs")
    }
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    StreamArgs::augment_args(super::population_definition().name("sentiment-corpus")
        .about("Score ordered complete documents with one resident INT4 sentiment model")
        .long_about("Stream bounded ordered {id,text,task_args?} NDJSON and exact {\"flush\":true} controls through one resident grouped-INT4 model. Every original document is losslessly partitioned and all chunks preflight before its first native forward. Retain each chunk's independent full-vocabulary affect distributions and original byte/scalar spans, plus the named scalar-weighted document aggregate. --defaults supplies {axes?,policy?}; absent defaults use all four axes and the existing descriptive policy. A non-null task_args replaces complete SentimentBatchArgs {axes,budget}; the run's policy, score space, EOS and model remain fixed. --max-axes is a hard capability ceiling for defaults and every override; choose compatible explicit defaults when narrowing it. Documents execute sequentially. Chunks are serial by default; --cohort-chunks shares native layers within a document and requires max_axes times the declared width <=64, with no clamping or fallback. All five native work flags bound the ENTIRE RUN, including failed admitted attempts, and never renew per document, chunk cohort or flush. Every document must finish completely before its success frame is published. Earlier completed documents remain valid after a later failure; require run_complete AND exit 0 for whole-run success. An exact drained flush resets only the ID epoch. Source conversion and independently pinned snapshots share this route. Requested snapshots are saved after fixed model/default/envelope admission and may remain after a later row failure. The chunk aggregate is uncalibrated and does not establish full-context equivalence, semantic accuracy or psychological measurement. No retained job, sampled sentiment, focus presets, prompt overrides, thinking, tools, --prefill-rows or --cohort-heads. Blocking IO observes cancellation cooperatively.")
        .mut_arg("cohort_chunks", |arg| arg.help("Share native layers across N chunks per document; --max-axes times N must be <=64"))
        .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; one bounded complete document per record"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded complete UTF-8 document bytes per record"))
        .mut_arg("max_map_result_bytes", |arg| arg.help("Maximum complete semantic document result; event/provenance framing is additional"))
        .mut_arg("max_checkpoints", |arg| arg.help("Nonrenewable cooperative checkpoint allowance for the ENTIRE stream"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream forward ceiling across every document, chunk and axis"))
        .mut_arg("max_projected_logits", |arg| arg.help("Whole-stream full-vocabulary scoring ceiling"))
        .mut_arg("max_attention_pairs", |arg| arg.help("Whole-stream attention-pair ceiling"))
        .mut_arg("max_dot_products", |arg| arg.help("Whole-stream native dot-product ceiling"))
        .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-stream multiply-accumulate ceiling")))
}

#[derive(Clone, Copy, Debug)]
pub(in crate::candidate_cli::int4) struct CorpusLimits {
    common: Int4Limits,
    document: MapLimits,
    transport: TransportLimits,
    output_bytes: u64,
    io_bytes: u64,
    cli_preparation_bytes: u64,
}

impl CorpusArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { document: MapArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }

    pub(in crate::candidate_cli::int4) fn validate(&self, args: &Int4Args) -> Result<CorpusLimits, CandidateError> {
        let s = &self.stream;
        if !(1..=4).contains(&s.max_axes)
            || self.document.mapping.cohort_chunks.is_some_and(|chunks|
                chunks.checked_mul(s.max_axes).is_none_or(|heads| heads == 0 || heads > 64)) {
            return Err(CandidateError::Arguments);
        }
        let document = self.document.validate_with_max_axes(args, s.max_axes)?;
        let mut common = document.common;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || s.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || s.defaults.as_ref().is_some_and(|path| path.as_os_str().is_empty() || path.as_os_str() == "-") {
            return Err(CandidateError::Arguments);
        }
        let framing = s.max_requests.checked_add(4)
            .and_then(|count| count.checked_mul(FRAME_ALLOWANCE)).ok_or(CandidateError::Arguments)?;
        // Transport capacity cannot multiply or replenish any native axis.
        let work = document.mapping.max_model_work;
        let transport = TransportLimits {
            max_line_bytes: s.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: document.envelope_bytes,
            max_output_bytes: s.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE,
            2 * IO_BUFFER_BYTES as u64])?;
        let cli_preparation_bytes = sum(&[document.cli_preparation_bytes, io_bytes])?;
        let corpus_buffers = corpus_buffer_bytes(transport, common.preparation_bytes, io_bytes)?;
        let metadata = retained_metadata_bytes(document.mapping.chunks.max_chunks)?;
        common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, corpus_buffers, metadata, io_bytes])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(CorpusLimits { common, document, transport, output_bytes: s.max_total_output_bytes,
            io_bytes, cli_preparation_bytes })
    }

    fn load_defaults(&self, text: Option<&str>, limits: CorpusLimits, args: &Int4Args)
        -> Result<(SentimentBatchArgs, SentimentPolicy), CandidateError> {
        let budget = self.document.budget_with_max_axes(limits.common, self.stream.max_axes)?;
        match shared::parse_defaults_with_limits(ScoreTask::Sentiment,
            self.document.scoring.classification_limits(args), limits.transport.max_document_bytes, text, budget)? {
            Defaults::Sentiment { args, policy } => {
                if args.axes.len() > self.stream.max_axes { return Err(CandidateError::Planning); }
                Ok((args, policy))
            },
            Defaults::Classify(_) | Defaults::Judge(_) => Err(CandidateError::Input),
        }
    }

    pub(in crate::candidate_cli::int4) fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }

    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W, diagnostics: &mut impl Write,
    ) -> ExitCode {
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(diagnostics, "fnlp candidate sentiment-corpus: {}", diagnostic(error));
                error.exit_code()
            }
        }
    }
}

fn diagnostic(error: CandidateError) -> &'static str {
    match error {
        CandidateError::Timeout => "cooperative corpus deadline expired; earlier completed frames may exist",
        CandidateError::Execution => "native corpus execution failed or was cancelled; earlier completed frames may exist",
        _ => error.message(),
    }
}

fn corpus_buffer_bytes(transport: TransportLimits, preparation: u64, io: u64) -> Result<u64, CandidateError> {
    let multiply = |value: u64, count: u64| value.checked_mul(count).ok_or(CandidateError::Arguments);
    sum(&[multiply(transport.max_line_bytes as u64, 8)?,
        multiply(transport.max_output_line_bytes as u64, 4)?,
        multiply(transport.max_epoch_ids as u64, 128)?, transport.max_epoch_id_bytes as u64,
        preparation, io])
}

fn retained_metadata_bytes(chunks: usize) -> Result<u64, CandidateError> {
    (chunks as u64).checked_mul(SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES).ok_or(CandidateError::Arguments)
}
