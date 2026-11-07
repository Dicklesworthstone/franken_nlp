//! Complete-source claim checking over one ordered resident-model stream.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork,
        judge::grouped_int4::FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

pub(super) const POLICY_BYTES: usize = 16_384;

#[derive(Args, Clone, Eq, PartialEq)]
struct StreamArgs {
    /// Fixed private FaithfulnessPolicy JSON; no claim, source or budget overrides.
    #[arg(long, value_name = "FILE", required = true)]
    policy: PathBuf,
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
        f.write_str("FaithfulnessCorpusArgs")
    }
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    StreamArgs::augment_args(super::definition().name("faithfulness-corpus")
        .about("Check ordered claim/source pairs with one resident complete-source INT4 judge")
        .long_about("Stream strict {id,text,task_args:{claim}} NDJSON and exact {\"flush\":true} controls through one resident grouped-INT4 model. text is the complete source; claim is required for each record. --policy supplies one strict FaithfulnessPolicy JSON object for the whole run. The complete source is losslessly partitioned with the exact claim repeated for each chunk. Every chunk and every actual ragged cohort window preflights before that document's first native forward. Retain every chunk's full-vocabulary NLI scores, contrary and uncertain evidence, original byte/scalar coordinates and the named complete-source reduction. Documents execute sequentially. --cohort-chunks explicitly shares native layers within a document; each actual window must fit 64 evidence heads, without changing the requested width or falling back. All five native work flags bound the ENTIRE RUN, including admitted failed attempts; flush controls reset only the ID epoch. Publish a success only after the entire document finishes. Earlier completed records remain valid after a later failure; require run_complete AND exit 0 for whole-run success. Source conversion and independently pinned snapshots share this route. Requested snapshots are saved after fixed model/policy/envelope admission and may remain after a later row fails. Policies are uncalibrated and evidence membership is not a correctness certificate or full-context equivalence claim. No per-row policy/budget overrides, generation, sampling, retained job, prompt overrides, thinking, tools, --prefill-rows or --cohort-heads. Blocking IO observes cancellation cooperatively.")
        .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; complete source text and a claim per record"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum combined decoded UTF-8 source and claim bytes per record"))
        .mut_arg("max_map_result_bytes", |arg| arg.help("Maximum complete semantic document result; event/provenance framing is additional"))
        .mut_arg("max_checkpoints", |arg| arg.help("Nonrenewable cooperative checkpoint allowance for the ENTIRE stream"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream forward ceiling across every source, chunk and evidence head"))
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
        let document = self.document.validate(args)?;
        let mut common = document.common;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || s.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || s.policy.as_os_str().is_empty() || s.policy.as_os_str() == "-" {
            return Err(CandidateError::Arguments);
        }
        let framing = s.max_requests.checked_add(4)
            .and_then(|count| count.checked_mul(FRAME_ALLOWANCE)).ok_or(CandidateError::Arguments)?;
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
        let buffers = corpus_buffer_bytes(transport, common.preparation_bytes, io_bytes)?;
        let metadata = retained_metadata_bytes(document.mapping.chunks.max_chunks)?;
        common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, buffers, metadata, io_bytes])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(CorpusLimits { common, document, transport, output_bytes: s.max_total_output_bytes,
            io_bytes, cli_preparation_bytes })
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
                let _ = writeln!(diagnostics, "fnlp candidate faithfulness-corpus: {}", diagnostic(error));
                error.exit_code()
            }
        }
    }
}

pub(super) fn parse_policy(text: &str) -> Result<FaithfulnessPolicy, CandidateError> {
    if text.len() > POLICY_BYTES { return Err(CandidateError::Input); }
    let value = crate::canonjson::parse_str_with_limits(text, crate::canonjson::ParseLimits {
        max_depth: 4, max_string_bytes: POLICY_BYTES,
    }).map_err(|_| CandidateError::Input)?;
    let policy: FaithfulnessPolicy = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    policy.validate().map_err(|_| CandidateError::Planning)?;
    Ok(policy)
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
    (chunks as u64).checked_mul(FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES).ok_or(CandidateError::Arguments)
}
