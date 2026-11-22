//! Explicit compact input for one complete INT8 source-map invocation.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};

#[derive(Args, Default)]
pub(in crate::candidate_cli) struct EncodingArgs {
    #[arg(skip)]
    pub input: input_encoding::InputEncodingArgs,
    /// Across every accepted/discarded probe and actual chunk plan, including hosted replanning.
    #[arg(long)]
    max_total_tokenizer_input_bytes: Option<u64>,
    /// Cumulative encoded tokens, separate from the native context per chunk.
    #[arg(long)]
    max_total_tokenizer_output_tokens: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_pair_checks: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_pair_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::candidate_cli) struct EncodingLimits {
    pub call: SafeBpeLimits,
    pub complete: SafeBpeWorkLimits,
}
impl EncodingLimits {
    pub(in crate::candidate_cli) fn policy(self) -> UntrustedEncoding { UntrustedEncoding::SafeBpeV1(self.call) }
    pub(in crate::candidate_cli) fn workspace_bytes(self) -> Result<u64, CandidateError> {
        self.complete.validate().map_err(|_| CandidateError::Arguments)?;
        self.call.requirements().map(|quote| quote.workspace_bytes + 1024)
            .map_err(|_| CandidateError::Arguments)
    }
}

impl EncodingArgs {
    pub(in crate::candidate_cli) fn limits(&self, command: &MapCommand)
        -> Result<Option<EncodingLimits>, CandidateError> {
        if self.input.is_default() {
            return if [self.max_total_tokenizer_input_bytes, self.max_total_tokenizer_output_tokens,
                self.max_total_tokenizer_pair_checks, self.max_total_tokenizer_pair_bytes]
                .iter().all(Option::is_none) { Ok(None) } else { Err(CandidateError::Arguments) };
        }
        // These dependent routes have different source/schema/evidence probes;
        // they must not silently inherit a ledger that does not price them.
        if !matches!(command.task.as_str(), "ner" | "keyphrases" | "summarize")
            || command.summary.synthesis.synthesize_summary {
            return Err(CandidateError::Arguments);
        }
        let bytes = command.max_chunk_bytes.unwrap_or(command.host.context_tokens)
            .min(command.host.max_input_bytes);
        let UntrustedEncoding::SafeBpeV1(call) = self.input.policy(bytes, bytes)?
            else { return Err(CandidateError::Arguments); };
        let default_bytes = (command.host.max_input_bytes as u64).checked_mul(8)
            .ok_or(CandidateError::Arguments)?;
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
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("BPE pair checks per encoding; --max-total-tokenizer-pair-checks also bounds the complete invocation"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("BPE examined pair bytes per encoding; --max-total-tokenizer-pair-bytes also bounds the complete invocation"))
        .after_long_help("--input-encoding safe-bpe-v1 supports complete NER, keyphrase and cited-summary maps, including exact --reduce-keyphrases and --reduce-summary. It preserves original source bytes and coordinates and excludes template-control tokens. All accepted/rejected sizing probes, CLI preflight and actual hosted chunk plans spend ONE nonrenewable tokenizer ledger. The four --max-total-tokenizer-* ceilings bound cumulative work; bytes and tokens default to eight times --max-input-bytes, pair checks to 268435456, and pair bytes to 4294967296. Per-call workspace is admitted before private IO and remains charged beside retained plans. Set --max-chunk-bytes to bound candidate bytes independently of native token capacity; omitted compact mode uses --context-tokens bytes. Tokenizer exhaustion fails the whole document without retry, byte fallback or partial success. Every ceiling binds execution identity; compact results report actual cumulative work. QA, schema extraction and neural synthesis do not accept compact input on this INT8 map route. Changed prompt tokens imply no identical-output, tokenizer-parity, quality or speed claim.")
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
pub(in crate::candidate_cli) struct EncodingReport {
    policy: UntrustedEncoding,
    complete_operation_limits: SafeBpeWorkLimits,
    work_used: crate::tokenizer::untrusted::SafeBpeWork,
}
#[cfg(feature = "asupersync-runtime")]
pub(in crate::candidate_cli) fn report(planner: &crate::tasks::source_planning::SourceTaskPlanner)
    -> Result<Option<EncodingReport>, CandidateError> {
    match planner.map_document_encoding_limits() {
        None => Ok(None),
        Some(complete_operation_limits) => Ok(Some(EncodingReport {
            policy: planner.document_encoding().clone(), complete_operation_limits,
            work_used: planner.map_document_encoding_work().map_err(|_| CandidateError::Execution)?
                .ok_or(CandidateError::Execution)?,
        })),
    }
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
pub(in crate::candidate_cli) struct EncodedResponse<'a, T: Serialize> {
    #[serde(flatten)]
    pub response: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<&'a EncodingReport>,
}
