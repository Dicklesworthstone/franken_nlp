//! One bounded tokenizer ledger covers original and edited-source redaction.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding};
pub(super) use crate::candidate_cli::int4::source_map::encoding::EncodingLimits;

#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct EncodingArgs {
    /// Total encoded bytes across original preflight, execution and verification.
    #[arg(long)]
    max_total_tokenizer_input_bytes: Option<u64>,
    /// Total untrusted tokens across every chunk probe and both detector stages.
    #[arg(long)]
    max_total_tokenizer_output_tokens: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_pair_checks: Option<u64>,
    #[arg(long)]
    max_total_tokenizer_pair_bytes: Option<u64>,
}

impl EncodingArgs {
    pub(super) fn limits(self, args: &Int4Args, mapping: &MappingFlags, result_bytes: usize)
        -> Result<Option<EncodingLimits>, CandidateError> {
        if args.input_encoding.is_default() {
            return if self == Self::default() { Ok(None) } else { Err(CandidateError::Arguments) };
        }
        // A sizing probe may exceed the final context. Count its exact tokens
        // so the lossless partitioner can shrink it, then enforce the real
        // prompt/context bound when compiling the accepted chunk.
        let bytes = mapping.max_ner_chunk_bytes.unwrap_or(args.context_tokens)
            .min(args.max_input_bytes.max(result_bytes));
        let UntrustedEncoding::SafeBpeV1(call) = args.input_encoding.policy(bytes, bytes)?
            else { return Err(CandidateError::Arguments); };
        // Edited placeholders/pseudonyms can expand the text. Both byte
        // envelopes contribute to the default complete-operation ceiling.
        let default_bytes = u64::try_from(args.max_input_bytes).ok()
            .and_then(|input| input.checked_add(result_bytes as u64))
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

pub(super) const HELP: &str = "--input-encoding safe-bpe-v1 preserves exact UTF-8 source bytes and excludes template-control tokens while compacting ordinary text. One nonrenewable tokenizer ledger covers the original preflight, every accepted/rejected chunk probe, execution replanning, and fresh detection of the actual edited text. --max-tokenizer-pair-checks and --max-tokenizer-pair-bytes bound each encoding call; --max-total-tokenizer-* bound the complete document operation. Total byte/token defaults are eight times the sum of the original and edited-result byte ceilings; total pair defaults are 268435456 checks and 4294967296 examined bytes. Exhaustion refuses the result without fallback or retry. Encoder workspace is reserved before private IO. Compact tokenization changes prompt tokens and makes no parity, quality or speed claim. Corpus/job commands create one admitted ledger per original document attempt, retaining it through both detector stages; flushes and chunks never renew it. Explicit resume may create a new invocation-local tokenizer ledger under the same authenticated policy; lifetime native/mask/attempt accounting is unchanged.";

pub(super) fn definition(command: clap::Command) -> clap::Command {
    input_encoding::definition(command)
        .after_long_help(HELP)
        .mut_arg("input_encoding", |arg| arg.help("Untrusted source policy; safe-bpe-v1 shares one tokenizer ledger across original and edited-source detection"))
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("BPE checks per encoding call; also bounded by --max-total-tokenizer-pair-checks"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("BPE examined bytes per encoding call; also bounded by --max-total-tokenizer-pair-bytes"))
}

#[cfg(feature = "asupersync-runtime")]
pub(super) fn configure(planner: crate::tasks::source_planning::SourceTaskPlanner,
    limits: Option<EncodingLimits>) -> Result<crate::tasks::source_planning::SourceTaskPlanner, CandidateError> {
    match limits {
        None => Ok(planner),
        Some(limits) => planner.with_map_document_encoding(limits.policy(), limits.complete)
            .map_err(|_| CandidateError::Planning),
    }
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
struct EncodingReport {
    policy: UntrustedEncoding,
    complete_document_limits: SafeBpeWorkLimits,
    budget_scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    work_used: Option<crate::tokenizer::untrusted::SafeBpeWork>,
}
#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
pub(super) struct Response<T: Serialize> {
    #[serde(flatten)]
    response: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<EncodingReport>,
}
#[cfg(feature = "asupersync-runtime")]
pub(super) fn population<T: Serialize>(response: T, limits: Option<EncodingLimits>) -> Response<T> {
    Response { response, input_encoding: limits.map(|limits| EncodingReport {
        policy: limits.policy(), complete_document_limits: limits.complete,
        budget_scope: "original_preflight_execution_and_edited_verification_per_document_attempt",
        work_used: None,
    }) }
}
#[cfg(feature = "asupersync-runtime")]
pub(super) fn completed<T: Serialize>(response: T,
    planner: &crate::tasks::source_planning::SourceTaskPlanner) -> Result<Response<T>, CandidateError> {
    let input_encoding = match planner.map_document_encoding_limits() {
        None => None,
        Some(complete_document_limits) => Some(EncodingReport {
            policy: planner.document_encoding().clone(), complete_document_limits,
            budget_scope: "original_preflight_execution_and_edited_verification",
            work_used: Some(planner.map_document_encoding_work().map_err(|_| CandidateError::Execution)?
                .ok_or(CandidateError::Execution)?),
        }),
    };
    Ok(Response { response, input_encoding })
}
