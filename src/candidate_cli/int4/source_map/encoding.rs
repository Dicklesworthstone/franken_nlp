//! One compact-tokenization ledger for a complete source-map invocation.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};
#[cfg(feature = "asupersync-runtime")]
use crate::tokenizer::untrusted::SafeBpeWork;

#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct EncodingArgs {
    /// Across ALL chunk probes and plan compilation, including host replanning.
    #[arg(long)]
    max_total_tokenizer_input_bytes: Option<u64>,
    /// Across ALL encodings, separate from each actual model prompt's cap.
    #[arg(long)]
    max_total_tokenizer_output_tokens: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_pair_checks: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_pair_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct EncodingLimits {
    pub call: SafeBpeLimits,
    pub complete: SafeBpeWorkLimits,
}
impl EncodingLimits {
    #[cfg(feature = "asupersync-runtime")]
    pub(super) fn policy(self) -> UntrustedEncoding { UntrustedEncoding::SafeBpeV1(self.call) }
    pub(super) fn workspace_bytes(self) -> Result<u64, CandidateError> {
        self.complete.validate().map_err(|_| CandidateError::Arguments)?;
        self.call.requirements().map(|quote| quote.workspace_bytes + 512)
            .map_err(|_| CandidateError::Arguments)
    }
}

impl EncodingArgs {
    pub(super) fn limits(self, args: &Int4Args, mapping: &MappingArgs,
        supported: bool, question: bool) -> Result<Option<EncodingLimits>, CandidateError> {
        if args.input_encoding.is_default() {
            return if self == Self::default() { Ok(None) } else { Err(CandidateError::Arguments) };
        }
        if !supported { return Err(CandidateError::Arguments); }
        // QA encodes the question, canonical passage manifest and source in
        // one request. Its per-call byte ceiling cannot be the source chunk
        // ceiling alone. Complete-document bytes still have their own bound.
        let bytes = if question { args.max_input_bytes.min(SafeBpeLimits::MAX_INPUT_BYTES) }
            else { mapping.max_chunk_bytes.unwrap_or(args.context_tokens).min(args.max_input_bytes) };
        // Probes must be able to count an oversized candidate so the existing
        // lossless partitioner can shrink it. Actual chunk plans separately
        // enforce the full prompt/context limit.
        let UntrustedEncoding::SafeBpeV1(call) = args.input_encoding.policy(bytes, bytes)?
            else { return Err(CandidateError::Arguments); };
        let default_bytes = u64::try_from(args.max_input_bytes).ok()
            .and_then(|bytes| bytes.checked_mul(8)).ok_or(CandidateError::Arguments)?;
        let complete = SafeBpeWorkLimits {
            max_input_bytes: self.max_total_tokenizer_input_bytes.unwrap_or(default_bytes),
            max_output_tokens: self.max_total_tokenizer_output_tokens.unwrap_or(default_bytes),
            max_pair_checks: self.max_total_tokenizer_pair_checks.unwrap_or(268_435_456),
            max_pair_bytes: self.max_total_tokenizer_pair_bytes.unwrap_or(4_294_967_296),
        };
        complete.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(Some(EncodingLimits { call, complete }))
    }
}

pub(super) fn definition(command: clap::Command) -> clap::Command {
    input_encoding::definition(command)
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("Safe-BPE pair checks per chunk encoding; the complete invocation also has --max-total-tokenizer-pair-checks"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("Safe-BPE pair bytes per chunk encoding; the complete invocation also has --max-total-tokenizer-pair-bytes"))
        .after_long_help("--input-encoding safe-bpe-v1 preserves exact document bytes and excludes every template control token while compacting ordinary text. NER, keyphrase, plain cited-summary and independent document-answer maps support this option, including --cohort-chunks and exact reducers. QA encodes the question, canonical passage manifest and source under one per-call ceiling bounded by --max-input-bytes and the encoder's one-MiB limit. All chunk-size probes, real chunk plans and the hosted repeat preflight consume ONE nonrenewable tokenizer ledger. --max-total-tokenizer-input-bytes and --max-total-tokenizer-output-tokens default to eight times --max-input-bytes; total pair checks default to 268435456 and total examined pair bytes to 4294967296. Exhaustion refuses the whole document without retry or a partial result. Per-call tokenizer workspace is admitted before private input/model IO. Policy and every ceiling bind prompt identity; successful output reports cumulative work separately. Neural synthesis/hierarchies, corpora and durable jobs remain byte-only.")
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
pub(super) struct EncodingReport {
    policy: UntrustedEncoding,
    complete_operation_limits: SafeBpeWorkLimits,
    work_used: SafeBpeWork,
}
#[cfg(feature = "asupersync-runtime")]
pub(super) fn report(planner: &crate::tasks::source_planning::SourceTaskPlanner)
    -> Result<Option<EncodingReport>, CandidateError> {
    match planner.map_document_encoding_limits() {
        None => Ok(None),
        Some(complete_operation_limits) => {
            let work_used = planner.map_document_encoding_work().map_err(|_| CandidateError::Execution)?
                .ok_or(CandidateError::Execution)?;
            Ok(Some(EncodingReport { policy: planner.document_encoding().clone(),
                complete_operation_limits, work_used }))
        },
    }
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
pub(super) struct EncodedResponse<'a, T: Serialize> {
    #[serde(flatten)]
    pub response: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<&'a EncodingReport>,
}
