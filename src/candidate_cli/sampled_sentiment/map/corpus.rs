//! Ordered complete-document samples with separate item and population work.
use super::*;
use crate::{
    batch::{BatchLimits, BatchWork},
    candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
};

#[derive(Args)]
struct StreamArgs {
    /// Nonempty records, including malformed input and flush controls.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    #[arg(long, default_value_t = 64)]
    max_input_mib: u64,
    /// Entire framed output, including failures, flushes and final completion.
    #[arg(long, default_value_t = 64)]
    max_output_mib: u64,
    #[arg(long, default_value_t = 4_194_304)]
    max_line_bytes: usize,
}

#[derive(Args)]
struct DocumentWorkArgs {
    /// Optional whole-document cap within the nonrenewable population allowance.
    #[arg(long)]
    max_document_forward_positions: Option<u64>,
    #[arg(long)]
    max_document_projected_logits: Option<u64>,
    #[arg(long)]
    max_document_attention_pairs: Option<u64>,
    #[arg(long)]
    max_document_dot_products: Option<u64>,
    #[arg(long)]
    max_document_multiply_accumulates: Option<u64>,
    #[arg(long)]
    max_document_mask_node_visits: Option<u64>,
}

impl DocumentWorkArgs {
    fn apply(self, command: &mut SampledSentimentCommand) -> Result<(), CandidateError> {
        fn tighten(value: Option<u64>, total: u64) -> Result<u64, CandidateError> {
            let value = value.unwrap_or(total);
            if value == 0 || value > total { Err(CandidateError::Arguments) } else { Ok(value) }
        }
        command.max_forward_positions = tighten(self.max_document_forward_positions, command.max_forward_positions)?;
        command.max_projected_logits = tighten(self.max_document_projected_logits, command.max_projected_logits)?;
        command.max_attention_pairs = tighten(self.max_document_attention_pairs, command.max_attention_pairs)?;
        command.max_dot_products = tighten(self.max_document_dot_products, command.max_dot_products)?;
        command.max_multiply_accumulates = tighten(self.max_document_multiply_accumulates, command.max_multiply_accumulates)?;
        command.max_mask_node_visits = tighten(self.max_document_mask_node_visits, command.max_mask_node_visits)?;
        Ok(())
    }
}

pub(in crate::candidate_cli) struct Command {
    document: SampledSentimentMapCommand,
    work: DocumentWorkArgs,
    stream: StreamArgs,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    StreamArgs::augment_args(DocumentWorkArgs::augment_args(super::definition()
        .name("sentiment-sampled-corpus")
        .about("Sample every source chunk of ordered documents on one resident INT8 model")
        .long_about("Read {id,text,task_args?} NDJSON records and exact {\"flush\":true} controls. Each text is a complete original UTF-8 document, losslessly partitioned to fit every requested axis prompt. Optional task_args.axes selects dimensions and task_args.budget may only tighten the task. All chunks, axes and independent score/explanation samples must complete before one document result is published. Results retain original source coordinates and complete chunk bundles, plus ordinal-paired document scores weighted by original Unicode scalar count. Stable record IDs address samples independently of stream order and flush epochs. The native and mask flags are nonrenewable whole-population ceilings; optional --max-document-* flags tighten each complete document. One cooperative deadline, checkpoint allowance and compact tokenizer ledger span this invocation. --compute-threads retains one numerical team across the population and composes with bounded --prefill-rows. Final success is withheld until every worker physically joins. A failed item makes the process exit nonzero. Explicit local candidate only; no automatic model discovery or download.")
        .mut_arg("input", |arg| arg.help("Complete-document NDJSON file, or '-' for stdin"))
        .mut_arg("max_input_bytes", |arg| arg.help("Original UTF-8 byte ceiling for each complete document"))
        .mut_arg("max_forward_positions", |arg| arg.help("Nonrenewable forward ceiling across the complete population"))
        .mut_arg("max_projected_logits", |arg| arg.help("Nonrenewable projected-logit ceiling across the complete population"))
        .mut_arg("max_attention_pairs", |arg| arg.help("Nonrenewable attention-pair ceiling across the complete population"))
        .mut_arg("max_dot_products", |arg| arg.help("Nonrenewable projection-dot ceiling across the complete population"))
        .mut_arg("max_multiply_accumulates", |arg| arg.help("Nonrenewable projection multiply-accumulate ceiling across the complete population"))
        .mut_arg("max_mask_node_visits", |arg| arg.help("Nonrenewable grammar-mask allowance across the complete population"))
        .mut_arg("max_tokenizer_input_bytes", |arg| arg.help("Cumulative untrusted bytes encoded across this invocation, including failed preparations; safe-bpe-v1 only"))
        .mut_arg("max_tokenizer_output_tokens", |arg| arg.help("Cumulative untrusted tokens encoded across this invocation; safe-bpe-v1 only"))
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("Cumulative BPE pair comparisons across this invocation; safe-bpe-v1 only"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("Cumulative examined BPE pair bytes across this invocation; safe-bpe-v1 only"))
        .after_long_help("Byte fallback is the default. Explicit safe-bpe-v1 preserves untrusted source bytes and excludes template-control tokens under ONE finite invocation ledger covering accepted and rejected chunk probes and every document preparation. Encoding work never renews at a record or flush. Per-document native and mask limits cannot exceed their population limits. Failed attempts retain all already reserved native and mask work. No document tail is omitted and a failed chunk produces no completed document. Generated-score variation is descriptive, not calibrated confidence; explanations are unverified model output. This candidate interface has no numerical, quality, throughput or release certification.")))
}

pub(in crate::candidate_cli) struct Envelope {
    pub transport: BatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
    pub transport_bytes: u64,
    pub max_model_work: Int8Work,
    pub max_mask_node_visits: u64,
}

impl Command {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { document: SampledSentimentMapCommand::from_matches(matches)?,
            work: DocumentWorkArgs::from_arg_matches(matches)?,
            stream: StreamArgs::from_arg_matches(matches)? })
    }

    pub(in crate::candidate_cli) fn admit(mut self)
        -> Result<(SampledSentimentMapCommand, MapLimits, Envelope), CandidateError> {
        let max_model_work = self.document.sampled.work_ceiling();
        let max_mask_node_visits = self.document.sampled.max_mask_node_visits;
        if !(1..=16 * 1024 * 1024).contains(&max_model_work.forward_positions)
            || max_model_work.projected_logits == 0 || max_model_work.attention_pairs == 0
            || max_model_work.projections.dot_products == 0
            || max_model_work.projections.multiply_accumulates == 0
            || !(1..=1_000_000_000_000).contains(&max_mask_node_visits) {
            return Err(CandidateError::Arguments);
        }
        self.work.apply(&mut self.document.sampled)?;
        let mut limits = self.document.validate()?;
        let s = self.stream;
        if !(1..=100_000).contains(&s.max_requests)
            || !(1..=16_384).contains(&s.max_input_mib)
            || !(1..=16_384).contains(&s.max_output_mib)
            || !(1..=4 * 1024 * 1024).contains(&s.max_line_bytes)
            || s.max_line_bytes < self.document.sampled.max_input_bytes {
            return Err(CandidateError::Arguments);
        }
        let output_bytes = s.max_output_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
        let framing = s.max_requests.checked_add(4).and_then(|n| n.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let transport = BatchLimits {
            max_line_bytes: s.max_line_bytes,
            max_document_bytes: self.document.sampled.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: s.max_input_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?,
            max_requests: s.max_requests, max_output_line_bytes: limits.envelope_bytes,
            max_output_bytes: output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: max_model_work.forward_positions,
                projected_logits: max_model_work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let line = u64::try_from(transport.max_output_line_bytes).map_err(|_| CandidateError::Arguments)?;
        let io_bytes = sum(&[line, FRAME_ALLOWANCE, 2 * IO_BUFFER_BYTES as u64])?;
        // Match the host's parser, canonical framing, duplicate-ID and IO
        // reservation in addition to the complete document's own live input.
        let transport_bytes = sum(&[
            line.checked_mul(4).ok_or(CandidateError::Arguments)?,
            (s.max_line_bytes as u64).checked_mul(8).ok_or(CandidateError::Arguments)?,
            (transport.max_epoch_ids as u64).checked_mul(128).ok_or(CandidateError::Arguments)?,
            transport.max_epoch_id_bytes as u64, io_bytes,
        ])?;
        limits.minimum_memory_bytes = limits.minimum_memory_bytes.checked_add(transport_bytes)
            .ok_or(CandidateError::Arguments)?;
        if limits.minimum_memory_bytes > limits.shared.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok((self.document, limits, Envelope { transport, output_bytes, io_bytes, transport_bytes,
            max_model_work, max_mask_node_visits }))
    }

    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        let result = self.admit().map_err(|_| Failure::usage("int8_sampled_sentiment_corpus_options"))
            .and_then(|(command, limits, envelope)| {
                #[cfg(feature = "asupersync-runtime")]
                { runtime::map::corpus::execute(command, limits, envelope, input, output) }
                #[cfg(not(feature = "asupersync-runtime"))]
                { let _ = (command, limits, envelope, input, output);
                    Err(Failure::usage("int8_sampled_sentiment_corpus_profile_unavailable")) }
            });
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => report_failure(failure, diagnostics),
        }
    }
}

#[cfg(test)] mod tests;
