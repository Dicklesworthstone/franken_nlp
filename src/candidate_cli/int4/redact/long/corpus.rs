//! One resident complete-document redaction stream with immutable policy/key scope.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
    tasks::redact::grouped_int4::corpus::Int4DocumentRedactionBatchConfig,
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
struct StreamFlags {
    /// Nonempty records, including malformed records and exact flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Complete stream bytes, including blank/oversized records and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// All emitted bytes, including provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// Encoded NDJSON bytes before LF, including JSON escaping and trailing CR.
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
    /// Tighten whole-stream authority; never renewed by documents or flush.
    #[arg(long)]
    max_corpus_forward_positions: Option<u64>,
    #[arg(long)]
    max_corpus_projected_logits: Option<u64>,
    #[arg(long)]
    max_corpus_attention_pairs: Option<u64>,
    #[arg(long)]
    max_corpus_dot_products: Option<u64>,
    #[arg(long)]
    max_corpus_multiply_accumulates: Option<u64>,
    #[arg(long)]
    max_corpus_mask_node_visits: Option<u64>,
}
#[derive(Clone, Eq, PartialEq)]
pub(in crate::candidate_cli::int4) struct CorpusArgs { document: DocumentArgs, stream: StreamFlags }
impl std::fmt::Debug for CorpusArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("RedactionCorpusArgs") }
}
#[derive(Clone, Copy)]
pub(in crate::candidate_cli::int4) struct CorpusLimits {
    pub common: Int4Limits,
    document: DocumentLimits,
    transport: TransportLimits,
    model_work: Int4Work,
    masks: u64,
    output_bytes: u64,
    io_bytes: u64,
}
pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    StreamFlags::augment_args(super::definition().name("redact-corpus")
        .about("Redact ordered complete-document NDJSON with one resident INT4 model and key scope")
        .long_about("Stream bounded {id,text,task_args?} NDJSON and exact {\"flush\":true} controls through one resident grouped-INT4 model and native allocation. Each document uses the complete-document redactor: whole-source rules, source-aligned NER chunks, one edit, and fresh transformed-text verification by default. The policy, NER types, binary key and namespace are fixed for the entire stream; task_args must be absent, null or an empty object. Completed document events remain valid if a later record fails. Require run_complete AND exit 0 for whole-run success. Per-document work/mask envelopes are charged to independent whole-corpus ceilings, including failed documents, and never renew at flush. IO/key ownership lasts through physical cleanup and writer flush. No retained job, per-item host invocation, rules-only fallback, automatic download or backend switch. NER chunk boundaries and model omissions can miss PII; clean scans and pseudonyms do not establish anonymity. Blocking IO observes cancellation cooperatively.")
        .mut_arg("input", |arg| arg.help("NDJSON file or '-' for stdin; binary key stdin requires a separate corpus file"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded COMPLETE original document bytes per record"))
        .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoint allowance for the ENTIRE corpus execution")))
}
impl CorpusArgs {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { document: DocumentArgs::from_matches(matches)?, stream: StreamFlags::from_arg_matches(matches)? })
    }
    pub(in crate::candidate_cli::int4) fn validate(&self, args: &Int4Args) -> Result<CorpusLimits, CandidateError> {
        let document = self.document.validate(args)?;
        let mut common = document.common;
        let s = self.stream;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_input_bytes)
            || !(1..=16 * 1024 * MIB).contains(&s.max_total_output_bytes)
            || s.max_line_bytes < args.max_input_bytes || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes) {
            return Err(CandidateError::Arguments);
        }
        let mut item = document.mapping.max_model_work;
        item.projected_logits -= item.projected_logits % NANBEIGE_VOCAB_SIZE as u64;
        let model_work = Int4Work {
            forward_positions: scaled_ceiling(item.forward_positions, s.max_requests, s.max_corpus_forward_positions)?,
            projected_logits: scaled_ceiling(item.projected_logits, s.max_requests, s.max_corpus_projected_logits)?,
            attention_pairs: scaled_ceiling(item.attention_pairs, s.max_requests, s.max_corpus_attention_pairs)?,
            projections: ProjectionWork {
                dot_products: scaled_ceiling(item.projections.dot_products, s.max_requests, s.max_corpus_dot_products)?,
                multiply_accumulates: scaled_ceiling(item.projections.multiply_accumulates, s.max_requests, s.max_corpus_multiply_accumulates)?,
            },
        };
        let masks = scaled_ceiling(document.mapping.max_mask_visits, s.max_requests, s.max_corpus_mask_node_visits)?;
        let frames = s.max_requests.checked_add(4).ok_or(CandidateError::Arguments)?;
        let transport = TransportLimits {
            max_line_bytes: s.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_total_input_bytes, max_requests: s.max_requests,
            max_output_line_bytes: common.result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: s.max_total_output_bytes.checked_sub(times(frames, FRAME_ALLOWANCE)?)
                .ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: model_work.forward_positions, projected_logits: model_work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE, IO_BUFFER_BYTES as u64 * 2])?;
        let corpus_buffers = sum(&[times(transport.max_line_bytes as u64, 8)?,
            times(transport.max_output_line_bytes as u64, 4)?, times(transport.max_epoch_ids as u64, 128)?,
            transport.max_epoch_id_bytes as u64, common.preparation_bytes, io_bytes])?;
        // Replace the standalone source/preparation capture with the complete
        // corpus buffer/preparation charge; keep native/map/edit/output/secret
        // reservations and the caller's simultaneously live preparation owner.
        let standalone_capture = sum(&[args.max_input_bytes as u64 + 1, common.preparation_bytes])?;
        common.minimum_memory_bytes = common.minimum_memory_bytes.checked_sub(standalone_capture)
            .and_then(|n| n.checked_add(corpus_buffers)).ok_or(CandidateError::Arguments)?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(CorpusLimits { common, document, transport, model_work, masks,
            output_bytes: s.max_total_output_bytes, io_bytes })
    }
    #[cfg(feature = "asupersync-runtime")]
    fn configuration(&self, args: &Int4Args, limits: CorpusLimits,
        identity: crate::execution_identity::ExecutionIdentity, ner: NerOptions) -> Int4DocumentRedactionBatchConfig {
        Int4DocumentRedactionBatchConfig { ner_identity: identity,
            detector: self.document.detector(args, limits.document, ner),
            request: self.document.request(args, limits.document),
            max_model_work: limits.model_work, max_mask_visits: limits.masks }
    }
    pub(in crate::candidate_cli::int4) fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
fn scaled_ceiling(item: u64, count: u64, explicit: Option<u64>) -> Result<u64, CandidateError> {
    if item == 0 || count == 0 { return Err(CandidateError::Arguments); }
    let total = match (item.checked_mul(count), explicit) {
        (Some(total), Some(cap)) => total.min(cap),
        (Some(total), None) => total,
        // An explicit finite ceiling remains below the mathematical product;
        // no saturated implicit allowance or wrapping counter is introduced.
        (None, Some(cap)) => cap,
        (None, None) => return Err(CandidateError::Arguments),
    };
    if total < item { return Err(CandidateError::Arguments); }
    Ok(total)
}
