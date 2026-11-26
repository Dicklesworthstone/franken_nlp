//! Complete-source claim checking on one resident INT8 model.
use super::*;
use crate::{
    batch::{BatchLimits, BatchWork},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
    tokenizer::untrusted::SafeBpeWorkLimits,
};

pub(in crate::candidate_cli) const POLICY_BYTES: usize = 16 * 1024;

#[derive(Args)]
pub(in crate::candidate_cli) struct TransportArgs {
    /// Fixed FaithfulnessPolicy JSON; each input record supplies only its claim.
    #[arg(long, value_name = "FILE", required = true)]
    pub(in crate::candidate_cli) policy: PathBuf,
    /// All nonempty records, including malformed records and flush commands.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// All transport bytes, including oversized records, blank lines and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All emitted bytes, including provenance, errors, flush and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// One NDJSON line before LF, including JSON escaping and a trailing CR.
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
    #[command(flatten)]
    encoding_work: jobs::EncodingWorkArgs,
}

pub(crate) struct FaithfulnessCorpusCommand {
    document: FaithfulnessMapCommand,
    pub(in crate::candidate_cli) transport: TransportArgs,
}

pub(in crate::candidate_cli) struct Admitted {
    pub document: MapLimits,
    pub transport: BatchLimits,
    pub max_tokenizer_work: Option<SafeBpeWorkLimits>,
    pub output_bytes: u64,
    pub io_bytes: u64,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    TransportArgs::augment_args(super::definition().name("faithfulness-corpus")
        .about("Check complete claim/source documents with one resident native INT8 judge")
        .long_about("Read ordered {id,text,task_args:{claim}} NDJSON. text is the complete original source; every record supplies an unchanged claim. --policy selects one strict FaithfulnessPolicy JSON object for the invocation. Losslessly partition each source into context-sized chunks and preserve every chunk judgment, contrary evidence, original byte/scalar coordinates and the complete-source reduction. A failed chunk produces no successful document result. One resident model serves the stream. All five native work ceilings, tokenizer totals and the original cooperative deadline span every record and failed attempt; flush resets only the duplicate-ID epoch. Earlier completed records remain valid after a later error. Require run_complete and exit 0 for whole-run success. Policies are uncalibrated; evidence membership does not establish factual correctness or single-context equivalence.")
        .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; strict {id,text,task_args:{claim}} records"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum combined decoded UTF-8 source and claim bytes per record; never truncate"))
        .mut_arg("max_checkpoints", |arg| arg.help("Original cooperative native checkpoint allowance for the complete stream"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream forward positions across every source, chunk and evidence head"))
        .mut_arg("max_projected_logits", |arg| arg.help("Whole-stream projected logits, including failed attempts"))
        .mut_arg("max_attention_pairs", |arg| arg.help("Whole-stream attention pairs, including failed attempts"))
        .mut_arg("max_dot_products", |arg| arg.help("Whole-stream native dot products, including failed attempts"))
        .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-stream multiply-accumulates, including failed attempts"))
        .after_long_help("--input-encoding safe-bpe-v1 preserves exact source/claim bytes while excluding template-control tokens. Per-document tokenizer caps cover sizing probes, all evidence heads and execution reconstruction; --max-total-tokenizer-* additionally bounds the invocation, including failed and cancelled preparation. Neither allowance resets at a chunk, head or flush. --prefill-rows and --compute-threads compose on the existing persistent INT8 team. Complete map, transport, encoder, output and worker memory is admitted before policy, population or model IO. One original source yields one complete document result with original evidence coordinates. Only faithfulness is accepted; no per-row policy/budget overrides, retained jobs, sampling, thinking, tools or cohort scheduling. Stdout contains candidate-framed NDJSON with complete_document provenance."))
}

impl FaithfulnessCorpusCommand {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { document: FaithfulnessMapCommand::from_matches(matches)?,
            transport: TransportArgs::from_arg_matches(matches)? })
    }

    pub(in crate::candidate_cli) fn validate(&self) -> Result<Admitted, CandidateError> {
        let document = self.document.validate()?;
        let args = &self.transport;
        if args.policy.as_os_str().is_empty() || args.policy.as_os_str() == "-"
            || !(1..=100_000).contains(&args.max_requests)
            || !(1..=16 * 1024 * MIB).contains(&args.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&args.max_total_output_bytes)
            || args.max_line_bytes < document.common.max_input_bytes
            || !(1..=4 * 1024 * 1024).contains(&args.max_line_bytes) {
            return Err(CandidateError::Arguments);
        }
        let max_tokenizer_work = args.encoding_work.for_document(&document.encoding)
            .map_err(|_| CandidateError::Arguments)?;
        if let (UntrustedEncoding::SafeBpeV1(call), Some(total)) = (&document.encoding, max_tokenizer_work) {
            if total.remaining_call(crate::tokenizer::untrusted::SafeBpeWork::default(), *call)
                .ok() != Some(*call) {
                return Err(CandidateError::Arguments);
            }
        }
        let framing = args.max_requests.checked_add(4)
            .and_then(|count| count.checked_mul(FRAME_ALLOWANCE)).ok_or(CandidateError::Arguments)?;
        let work = document.mapping.max_model_work;
        let transport = BatchLimits {
            max_line_bytes: args.max_line_bytes, max_document_bytes: document.common.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: args.max_total_input_bytes, max_requests: args.max_requests,
            max_output_line_bytes: document.envelope_bytes,
            max_output_bytes: args.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = (document.envelope_bytes as u64).checked_add(FRAME_ALLOWANCE)
            .and_then(|bytes| bytes.checked_add(2 * IO_BUFFER_BYTES as u64)).ok_or(CandidateError::Arguments)?;
        let admitted = Admitted { document, transport, max_tokenizer_work,
            output_bytes: args.max_total_output_bytes, io_bytes };
        #[cfg(feature = "asupersync-runtime")]
        runtime::faithfulness_map::corpus::admit(&admitted)?;
        Ok(admitted)
    }

    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W, diagnostics: &mut impl Write,
    ) -> ExitCode {
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(diagnostics, "fnlp candidate faithfulness-corpus: {}", error.message());
                error.exit_code()
            }
        }
    }

    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W,
    ) -> Result<(), CandidateError> {
        let admitted = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::faithfulness_map::corpus::execute(self, admitted, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, admitted, input, output); Err(CandidateError::Unavailable) }
    }
}

pub(in crate::candidate_cli) fn parse_policy(text: &str) -> Result<FaithfulnessPolicy, CandidateError> {
    if text.len() > POLICY_BYTES { return Err(CandidateError::Input); }
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
        max_depth: 4, max_string_bytes: POLICY_BYTES,
    }).map_err(|_| CandidateError::Input)?;
    let policy: FaithfulnessPolicy = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    policy.validate().map_err(|_| CandidateError::Planning)?;
    Ok(policy)
}

#[cfg(test)]
pub(in crate::candidate_cli) mod tests;
