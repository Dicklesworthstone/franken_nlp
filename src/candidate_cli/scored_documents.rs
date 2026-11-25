//! Complete-document finite-scoring streams on one resident INT8 model.
use super::*;
use crate::{
    batch::{BatchLimits, BatchWork},
    native_engine::strict_int8::{Int8Work, prefill::Int8PrefillLimits},
    tasks::{
        classify::{ClassificationLimits, quantized::long::Int8ClassificationMapLimits},
        sentiment::{SentimentLimits, quantized::long::Int8SentimentMapLimits},
        ir::TaskBudget,
    },
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};
use super::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES};

#[derive(Args)]
pub(super) struct TransportArgs {
    /// Fixed labels/mode/policy or sentiment axes/policy; never document or budget.
    #[arg(long, value_name = "FILE")]
    pub defaults: Option<PathBuf>,
    /// Whole-run nonempty records, including malformed input and flush commands.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    #[arg(long, default_value_t = 1024)]
    max_input_mib: u64,
    /// Includes candidate provenance, document errors, flush and terminal frames.
    #[arg(long, default_value_t = 1024)]
    max_output_mib: u64,
    /// NDJSON record bytes before LF, including escaped source and task settings.
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
    #[command(flatten)]
    encoding_work: jobs::EncodingWorkArgs,
}

pub(super) enum Task {
    Classify(classify_map::ClassifyMapCommand),
    Sentiment(sentiment_map::SentimentMapCommand),
}

pub(crate) struct DocumentCorpusCommand {
    pub(super) task: Task,
    pub(super) transport: TransportArgs,
}

pub(super) enum Mapping {
    Classify { planning: ClassificationLimits, mapping: Int8ClassificationMapLimits },
    Sentiment { planning: SentimentLimits, mapping: Int8SentimentMapLimits },
}
impl Mapping {
    pub(super) fn work(&self) -> Int8Work {
        match self {
            Self::Classify { mapping, .. } => mapping.max_model_work,
            Self::Sentiment { mapping, .. } => mapping.max_model_work,
        }
    }
    pub(super) fn task_name(&self) -> &'static str {
        match self { Self::Classify { .. } => "classify", Self::Sentiment { .. } => "sentiment" }
    }
}

pub(super) struct Admitted {
    pub common: CandidateArgs,
    pub limits: Limits,
    pub task: TaskBudget,
    pub mapping: Mapping,
    pub encoding: UntrustedEncoding,
    pub max_tokenizer_work: Option<SafeBpeWorkLimits>,
    pub prefill: Option<Int8PrefillLimits>,
    pub transport: BatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
}

pub(super) fn definitions() -> Vec<clap::Command> {
    [
        (classify_map::definition(), "classify-corpus",
            "Classify complete NDJSON documents with one resident INT8 model"),
        (sentiment_map::definition(), "sentiment-corpus",
            "Score complete NDJSON documents with one resident INT8 model"),
    ].into_iter().map(|(command, name, about)| {
        TransportArgs::augment_args(command.name(name).about(about)
            .long_about("Read bounded ordered {id,text,task_args?} NDJSON. Each text is a complete source document, losslessly partitioned into context-sized chunks. Retain every completed chunk with original byte/scalar coordinates and the source-weighted aggregate. Classification supports exclusive and independent multi-label decisions; sentiment retains independent affect dimensions. These are descriptive uncalibrated mixtures, not a single-context-equivalence claim. One resident model and native context serve the entire stream. All five native work ceilings and the original invocation deadline remain cumulative across documents, failed execution and flush. A failed chunk produces no successful document result. Earlier completed records remain valid; any failed record makes the process exit nonzero.")
            .mut_arg("input", |arg| arg.help("NDJSON {id,text,task_args?}; '-' streams stdin without whole-file buffering"))
            .mut_arg("max_input_bytes", |arg| arg.help("Maximum complete source bytes per record; never a truncation limit"))
            .mut_arg("max_result_bytes", |arg| arg.help("Maximum complete chunk-result bytes; --max-map-result-bytes separately caps the complete document"))
            .mut_arg("compute_threads", |arg| arg.help("Coordinator plus persistent INT8 workers for the entire stream, 1..=64; terminal completion follows physical worker join"))
            .mut_arg("max_forward_positions", |arg| arg.help("Whole-invocation forward positions across all records, chunks and heads; never renewed by flush"))
            .mut_arg("max_projected_logits", |arg| arg.help("Whole-invocation projected logits across all records, chunks and heads"))
            .mut_arg("max_attention_pairs", |arg| arg.help("Whole-invocation attention pairs, including failed execution attempts"))
            .mut_arg("max_dot_products", |arg| arg.help("Whole-invocation projection dot products, including failed execution attempts"))
            .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-invocation projection multiply-accumulates, including failed execution attempts"))
            .mut_arg("max_tokenizer_input_bytes", |arg| arg.help("Cumulative encoded bytes per complete document, including partition probes and reconstruction; safe-bpe-v1 only"))
            .mut_arg("max_tokenizer_output_tokens", |arg| arg.help("Cumulative untrusted tokens per complete document, including probes and reconstruction; separate from native context"))
            .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("Cumulative BPE pair checks per complete document, including probes and reconstruction; safe-bpe-v1 only"))
            .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("Cumulative BPE examined-pair bytes per complete document, including probes and reconstruction; safe-bpe-v1 only"))
            .after_long_help("Use --defaults FILE for shared classification labels/mode/policy or sentiment axes/policy. Per-record task_args use the existing typed batch schema and may only tighten the admitted task budget. Without classification defaults, every record supplies complete task_args with labels, mode, policy and budget. Supplied task_args replaces the entire defaults object. Sentiment defaults to all four affect axes. --input-encoding safe-bpe-v1 preserves exact source/label bytes and excludes template controls. Per-document tokenizer limits cover sizing and reconstruction; --max-total-tokenizer-* additionally bounds the entire invocation, including rejected preparations, and never resets at flush. Native work is not multiplied by --max-requests. --prefill-rows and --compute-threads compose on one persistent team with complete memory admission before private IO. No input/output retention, sampling, thinking, tools or cohort schedule is introduced. All events retain explicit non-authoritative candidate provenance."))
    }).collect()
}

impl DocumentCorpusCommand {
    pub(super) fn from_matches(name: &str, matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let task = match name {
            "classify-corpus" => Task::Classify(classify_map::ClassifyMapCommand::from_matches(matches)?),
            "sentiment-corpus" => Task::Sentiment(sentiment_map::SentimentMapCommand::from_matches(matches)?),
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "complete-document scoring task required")),
        };
        Ok(Self { task, transport: TransportArgs::from_arg_matches(matches)? })
    }

    fn scoring(&self) -> &scored::ScoredArgs {
        match &self.task { Task::Classify(command) => &command.scoring, Task::Sentiment(command) => &command.args }
    }

    pub(super) fn parse_defaults(&self, json: Option<&str>, task: TaskBudget)
        -> Result<scored_batch::Defaults, CandidateError> {
        let kind = match &self.task {
            Task::Classify(_) => scored_batch::ScoreTask::Classify,
            Task::Sentiment(_) => scored_batch::ScoreTask::Sentiment,
        };
        scored_batch::parse_defaults(kind, self.scoring(), json, task)
    }

    pub(super) fn validate(&self) -> Result<Admitted, CandidateError> {
        let (common, limits, task, mapping, encoding, prefill, output_line_bytes) = match &self.task {
            Task::Classify(command) => {
                let admitted = command.validate()?;
                (admitted.common, admitted.limits, admitted.task,
                    Mapping::Classify { planning: admitted.planning, mapping: admitted.mapping },
                    admitted.encoding, admitted.prefill, admitted.envelope_bytes)
            }
            Task::Sentiment(command) => {
                let admitted = command.validate()?;
                (admitted.common, admitted.shared, admitted.task,
                    Mapping::Sentiment { planning: command.args.sentiment_limits(), mapping: admitted.mapping },
                    admitted.encoding, admitted.prefill, admitted.envelope_bytes)
            }
        };
        let args = &self.transport;
        if !(1..=100_000).contains(&args.max_requests)
            || args.max_line_bytes < common.max_input_bytes || args.max_line_bytes > 4 * 1024 * 1024
            || !(1..=1024 * 1024).contains(&args.max_input_mib)
            || !(1..=1024 * 1024).contains(&args.max_output_mib)
            || args.defaults.as_ref().is_some_and(|path| path.as_os_str().is_empty() || path.as_os_str() == "-") {
            return Err(CandidateError::Arguments);
        }
        let max_tokenizer_work = args.encoding_work.for_document(&encoding)
            .map_err(|_| CandidateError::Arguments)?;
        // Classification admits its complete fixed per-preparation envelope
        // before spending compact work. Reject a smaller invocation envelope
        // here, before inspecting even model metadata or defaults.
        if let (Mapping::Classify { .. }, UntrustedEncoding::SafeBpeV1(call), Some(total)) =
            (&mapping, &encoding, max_tokenizer_work) {
            if total.remaining_call(crate::tokenizer::untrusted::SafeBpeWork::default(), *call)
                .ok() != Some(*call) {
                return Err(CandidateError::Arguments);
            }
        }
        let output_bytes = args.max_output_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
        let framing = args.max_requests.checked_add(4)
            .and_then(|n| n.checked_mul(FRAME_ALLOWANCE)).ok_or(CandidateError::Arguments)?;
        let work = mapping.work();
        let transport = BatchLimits {
            max_line_bytes: args.max_line_bytes, max_document_bytes: common.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: args.max_input_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?,
            max_requests: args.max_requests, max_output_line_bytes: output_line_bytes,
            max_output_bytes: output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = (output_line_bytes as u64).checked_add(FRAME_ALLOWANCE)
            .and_then(|n| n.checked_add(IO_BUFFER_BYTES as u64 * 2)).ok_or(CandidateError::Arguments)?;
        let admitted = Admitted { common, limits, task, mapping, encoding,
            max_tokenizer_work, prefill, transport, output_bytes, io_bytes };
        #[cfg(feature = "asupersync-runtime")]
        runtime::scored_documents::admit(&admitted)?;
        Ok(admitted)
    }

    pub(super) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W, diagnostics: &mut impl Write,
    ) -> ExitCode {
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(diagnostics, "fnlp candidate document-corpus: {}", error.message());
                error.exit_code()
            }
        }
    }

    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W,
    ) -> Result<(), CandidateError> {
        let admitted = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::scored_documents::execute(self, admitted, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, admitted, input, output); Err(CandidateError::Unavailable) }
    }
}

#[cfg(test)]
pub(super) mod tests;
