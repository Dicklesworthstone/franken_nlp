//! Ordered INT4 source-task streams with finite whole-run resource ceilings.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork, source::{SourceBatchArgs, SourceMaskBudget,
        grouped_int4::{Int4SourceBatchLimits, MAX_SOURCE_ARGUMENT_BYTES}}},
    candidate_cli::{batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES}, source::Kind},
    native_engine::{constrained_int4, strict_int4::Int4Work},
    tasks::{keyphrases::KeyphraseOptions, ner::NerOptions, summarize::SummaryOptions, ir::TaskBudget},
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BatchArgs { pub kind: Kind, pub source: source::SourceArgs, stream: StreamArgs }
#[derive(Args, Clone, Eq, PartialEq)]
struct StreamArgs {
    /// Complete SourceBatchArgs JSON. Without it QA requires per-record task_args.
    #[arg(long, value_name = "FILE")]
    defaults: Option<PathBuf>,
    /// Whole-run nonempty records, including malformed input and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Whole input stream, including oversized records, blank lines and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All emitted bytes, including candidate provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// Maximum NDJSON bytes before LF, including JSON syntax and trailing CR.
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
}
impl std::fmt::Debug for StreamArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("SourceStreamArgs") }
}
pub(super) fn definitions() -> Vec<clap::Command> {
    [(Kind::Ner, "ner-batch"), (Kind::Keyphrases, "keyphrases-batch"),
        (Kind::Summarize, "summarize-batch"), (Kind::Answer, "answer-batch")]
        .into_iter().map(|(kind, name)| StreamArgs::augment_args(source::definition(kind).name(name)
            .about("Run an ordered source-task NDJSON stream with one resident grouped-INT4 model")
            .long_about("Run one fixed source task over ordered {id,text,task_args?} NDJSON and exact {\"flush\":true} controls. NER, keyphrases and summarize use shipped options unless --defaults or per-record typed task_args supplies explicit bounded options. For answer, text is the question and task_args supplies separately typed passages, options and budget; no passage set is invented or reused implicitly. Defaults files use complete SourceBatchArgs JSON, including its task selector and budget. A record cannot switch tasks or raise the CLI ceilings. Every completed event carries candidate provenance. Require run_complete AND exit 0 for whole-run success; earlier completed records remain valid if a later record fails. Native work, masks, checkpoints and transport ceilings span all epochs. Execution is serial item-local constrained decoding. Structural citations do not certify semantic support; failed inference is never abstention. Input/writer ownership survives cancellation until hosted cleanup.")
            .mut_arg("options", |arg| arg.hide(true))
            .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; each source record is bounded and read incrementally"))
            .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded document/question and passage input bytes per record"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoint allowance for the ENTIRE corpus execution"))))
        .collect()
}
#[derive(Clone, Copy, Debug)]
pub(super) struct BatchLimits {
    pub common: Int4Limits,
    pub transport: TransportLimits,
    pub native: Int4SourceBatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
    pub preparation_reserve_bytes: u64,
}
impl BatchArgs {
    pub(super) fn from_matches(kind: Kind, matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if matches.value_source("options") == Some(clap::parser::ValueSource::CommandLine) {
            return Err(clap::Error::raw(clap::error::ErrorKind::ArgumentConflict,
                "source-task batches use complete --defaults or per-record task_args, not --options"));
        }
        Ok(Self { kind, source: source::SourceArgs::from_matches(matches)?, stream: StreamArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<BatchLimits, CandidateError> {
        let stream = &self.stream;
        if !(1..=100_000).contains(&stream.max_requests)
            || stream.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&stream.max_line_bytes)
            || !(1..=16 * 1024 * MIB).contains(&stream.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&stream.max_total_output_bytes)
            || self.source.options_path().is_some()
            || stream.defaults.as_ref().is_some_and(|path| path.as_os_str().is_empty() || path.as_os_str() == "-") {
            return Err(CandidateError::Arguments);
        }
        let mut common = self.source.validate(args)?;
        let one = constrained_int4::planned_work(args.group_width, common.max_prompt_tokens, args.max_new_tokens)
            .map_err(|_| CandidateError::Arguments)?;
        // Scale the actual independent per-item attention triangles. Never
        // price a fictitious concatenated sequence or use INT8 group counts.
        let work = scale_work(one, stream.max_requests)?;
        let native = Int4SourceBatchLimits { max_model_work: work, masks: SourceMaskBudget {
            per_mask: self.source.masks(), max_visits_per_item: self.source.mask_visits(),
            max_visits_per_run: times(self.source.mask_visits(), stream.max_requests)?,
        } };
        let frames = stream.max_requests.checked_add(4).ok_or(CandidateError::Arguments)?;
        let transport = TransportLimits {
            max_line_bytes: stream.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: stream.max_total_input_bytes, max_requests: stream.max_requests,
            max_output_line_bytes: common.result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: stream.max_total_output_bytes.checked_sub(times(frames, FRAME_ALLOWANCE)?)
                .ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        native.validate().map_err(|_| CandidateError::Arguments)?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE, IO_BUFFER_BYTES as u64])?;
        let corpus_reservation = sum(&[times(transport.max_line_bytes as u64, 8)?,
            times(transport.max_output_line_bytes as u64, 4)?, times(transport.max_epoch_ids as u64, 128)?,
            transport.max_epoch_id_bytes as u64, preparation_reserve_bytes, io_bytes])?;
        common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, corpus_reservation])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(BatchLimits { common, transport, native, output_bytes: stream.max_total_output_bytes,
            io_bytes, preparation_reserve_bytes })
    }
    pub(super) fn load_defaults(&self, text: Option<&str>, ceiling: TaskBudget, max_input_bytes: usize)
        -> Result<Option<SourceBatchArgs>, CandidateError> {
        let defaults = match text {
            Some(text) => {
                if text.len() > MAX_SOURCE_ARGUMENT_BYTES { return Err(CandidateError::Input); }
                let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
                    max_depth: 16, max_string_bytes: MAX_SOURCE_ARGUMENT_BYTES,
                }).map_err(|_| CandidateError::Input)?;
                Some(serde_json::from_value::<SourceBatchArgs>(value).map_err(|_| CandidateError::Input)?)
            }
            None => match self.kind {
                Kind::Ner => Some(SourceBatchArgs::Ner { options: NerOptions::default(), budget: ceiling }),
                Kind::Keyphrases => Some(SourceBatchArgs::Keyphrases { options: KeyphraseOptions::default(), budget: ceiling }),
                Kind::Summarize => Some(SourceBatchArgs::Summarize { options: SummaryOptions::default(), budget: ceiling }),
                Kind::Answer => None,
            },
        };
        if let Some(defaults) = &defaults { self.validate_defaults(defaults, ceiling, max_input_bytes)?; }
        Ok(defaults)
    }
    fn validate_defaults(&self, defaults: &SourceBatchArgs, ceiling: TaskBudget, max_input_bytes: usize)
        -> Result<(), CandidateError> {
        let (kind, budget) = match defaults {
            SourceBatchArgs::Ner { options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?; (Kind::Ner, budget)
            }
            SourceBatchArgs::Keyphrases { options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?; (Kind::Keyphrases, budget)
            }
            SourceBatchArgs::Summarize { options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?; (Kind::Summarize, budget)
            }
            SourceBatchArgs::Answer { passages, options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?;
                if passages.is_empty() || passages.len() > crate::candidate_cli::source::MAX_PASSAGES {
                    return Err(CandidateError::Input);
                }
                let mut ids = std::collections::BTreeSet::new();
                let mut bytes = 0_usize;
                for passage in passages {
                    if passage.id.is_empty() || passage.id.len() > 128 || passage.id.chars().any(char::is_control)
                        || !ids.insert(passage.id.as_str()) { return Err(CandidateError::Input); }
                    bytes = bytes.checked_add(passage.id.len()).and_then(|n| n.checked_add(passage.text.len()))
                        .filter(|&n| n <= max_input_bytes).ok_or(CandidateError::Input)?;
                }
                (Kind::Answer, budget)
            }
        };
        budget.validate().map_err(|_| CandidateError::Planning)?;
        if kind != self.kind || budget.max_input_tokens > ceiling.max_input_tokens
            || budget.max_output_tokens > ceiling.max_output_tokens || budget.max_output_bytes > ceiling.max_output_bytes
            || budget.max_grammar_states > ceiling.max_grammar_states || budget.max_kv_bytes != ceiling.max_kv_bytes {
            return Err(CandidateError::Planning);
        }
        Ok(())
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
