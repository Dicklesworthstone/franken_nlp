//! Owned redaction NDJSON mode. Document policy/key flags are reused unchanged.
use super::*;
use crate::{
    batch::{BatchLimits, BatchWork},
    native_engine::{constrained_int8, lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::Int8Work},
    tasks::redact::quantized::Int8RedactionConfig,
};
use crate::candidate_cli::batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES};

#[derive(Args)]
pub(in crate::candidate_cli) struct CorpusArgs {
    /// Read bounded {id,text,task_args?} NDJSON with one resident model and key scope.
    /// --chunked applies to each document; records cannot override the policy.
    #[arg(long)]
    pub ndjson: bool,
    /// Nonempty records, including malformed records and flush commands (default 1000).
    #[arg(long, requires = "ndjson")]
    max_requests: Option<u64>,
    /// Whole-stream input bytes in MiB, including blank/oversized lines (default 1024).
    #[arg(long, requires = "ndjson")]
    max_input_mib: Option<u64>,
    /// All live output bytes, or combined retained spool/materialization caps (default 1024 MiB).
    #[arg(long, requires = "ndjson")]
    max_output_mib: Option<u64>,
    /// NDJSON bytes before LF, including JSON syntax/escaping (default 1048576).
    #[arg(long, requires = "ndjson")]
    max_line_bytes: Option<usize>,
    /// Tighten the derived whole-stream ceiling; never renewed by a record or flush.
    #[arg(long, requires = "ndjson")]
    max_corpus_forward_positions: Option<u64>,
    #[arg(long, requires = "ndjson")]
    max_corpus_projected_logits: Option<u64>,
    #[arg(long, requires = "ndjson")]
    max_corpus_attention_pairs: Option<u64>,
    #[arg(long, requires = "ndjson")]
    max_corpus_dot_products: Option<u64>,
    #[arg(long, requires = "ndjson")]
    max_corpus_multiply_accumulates: Option<u64>,
    #[arg(long, requires = "ndjson")]
    max_corpus_mask_node_visits: Option<u64>,
}
#[derive(Clone, Copy)]
pub(in crate::candidate_cli) struct CorpusEnvelope {
    pub transport: BatchLimits,
    pub max_model_work: Int8Work,
    pub max_mask_visits: u64,
    pub output_bytes: u64,
    pub io_bytes: u64,
}
impl CorpusArgs {
    pub(super) fn validate(&self, command: &RedactCommand, limits: Limits) -> Result<(), CandidateError> {
        if self.ndjson { return self.envelope(command, limits).map(|_| ()); }
        if self.max_requests.is_some() || self.max_input_mib.is_some() || self.max_output_mib.is_some()
            || self.max_line_bytes.is_some() || self.max_corpus_forward_positions.is_some()
            || self.max_corpus_projected_logits.is_some() || self.max_corpus_attention_pairs.is_some()
            || self.max_corpus_dot_products.is_some() || self.max_corpus_multiply_accumulates.is_some()
            || self.max_corpus_mask_node_visits.is_some() { return Err(CandidateError::Arguments); }
        Ok(())
    }
    pub(in crate::candidate_cli) fn envelope(&self, command: &RedactCommand, limits: Limits)
        -> Result<CorpusEnvelope, CandidateError> {
        let count = self.max_requests.unwrap_or(1000);
        let input_mib = self.max_input_mib.unwrap_or(1024);
        let output_mib = self.max_output_mib.unwrap_or(1024);
        let line_bytes = self.max_line_bytes.unwrap_or(1_048_576);
        if !self.ndjson || !(1..=100_000).contains(&count)
            || !(1..=1024 * 1024).contains(&input_mib) || !(1..=1024 * 1024).contains(&output_mib)
            || line_bytes < command.host.max_input_bytes || line_bytes > 4 * 1024 * 1024 {
            return Err(CandidateError::Arguments);
        }
        let (mut item, masks) = if command.long.chunked {
            let d = command.long.config(&command.host, limits, NerOptions::default())?;
            (d.mapping.max_model_work, d.mapping.max_mask_visits)
        } else {
            let d = short_detector(command, limits, NerOptions::default())?;
            (d.max_model_work, d.max_mask_visits)
        };
        // The pinned constrained driver projects complete vocabulary rows.
        item.projected_logits -= item.projected_logits % NANBEIGE_VOCAB_SIZE as u64;
        if item.projected_logits == 0 { return Err(CandidateError::Arguments); }
        let mut work = scale_work(item, count)?;
        let mut mask_total = masks.checked_mul(count).ok_or(CandidateError::Arguments)?;
        tighten(&mut work.forward_positions, self.max_corpus_forward_positions, item.forward_positions)?;
        tighten(&mut work.projected_logits, self.max_corpus_projected_logits, item.projected_logits)?;
        tighten(&mut work.attention_pairs, self.max_corpus_attention_pairs, item.attention_pairs)?;
        tighten(&mut work.projections.dot_products, self.max_corpus_dot_products, item.projections.dot_products)?;
        tighten(&mut work.projections.multiply_accumulates, self.max_corpus_multiply_accumulates, item.projections.multiply_accumulates)?;
        tighten(&mut mask_total, self.max_corpus_mask_node_visits, masks)?;
        let output_bytes = output_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
        // Start, up to N request events, EOF flush, terminal, plus a read-failure
        // frame at the next record. Native and outer sinks BOTH enforce bytes.
        let framing = count.checked_add(4).and_then(|n| n.checked_mul(FRAME_ALLOWANCE)).ok_or(CandidateError::Arguments)?;
        let inner_bytes = output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?;
        let transport = BatchLimits {
            max_line_bytes: line_bytes, max_document_bytes: command.host.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: input_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?, max_requests: count,
            max_output_line_bytes: command.host.max_result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: inner_bytes,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = (transport.max_output_line_bytes as u64).checked_add(FRAME_ALLOWANCE)
            .and_then(|n| n.checked_add(IO_BUFFER_BYTES as u64 * 2)).ok_or(CandidateError::Arguments)?;
        Ok(CorpusEnvelope { transport, max_model_work: work, max_mask_visits: mask_total, output_bytes, io_bytes })
    }
}
pub(in crate::candidate_cli) fn short_detector(command: &RedactCommand, limits: Limits, ner: NerOptions)
    -> Result<Int8RedactionConfig, CandidateError> {
    let host = &command.host;
    let stages = 1 + u64::from(!command.no_verify);
    let one = constrained_int8::planned_work(host.context_tokens - host.max_new_tokens, host.max_new_tokens)
        .map_err(|_| CandidateError::Arguments)?;
    let mut planning = host.planning(); planning.max_input_bytes = host.max_input_bytes.max(host.max_result_bytes);
    Ok(Int8RedactionConfig { ner, per_pass: host.task_budget(limits), planning,
        max_model_work: scale_work(one, stages)?, mask_limits: host.masks(), mask_visits_per_pass: host.max_mask_node_visits,
        max_mask_visits: host.max_mask_node_visits.checked_mul(stages).ok_or(CandidateError::Arguments)?,
        max_result_bytes: host.max_result_bytes as u64 })
}
fn scale_work(mut work: Int8Work, count: u64) -> Result<Int8Work, CandidateError> {
    for value in [&mut work.forward_positions, &mut work.projected_logits, &mut work.attention_pairs,
        &mut work.projections.dot_products, &mut work.projections.multiply_accumulates] {
        *value = value.checked_mul(count).ok_or(CandidateError::Arguments)?;
    }
    Ok(work)
}
fn tighten(total: &mut u64, cap: Option<u64>, item: u64) -> Result<(), CandidateError> {
    if let Some(cap) = cap { *total = (*total).min(cap); }
    if item == 0 || *total < item { return Err(CandidateError::Arguments); }
    Ok(())
}
impl RedactCommand {
    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(self,
        input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        let retained = self.retention.store_results;
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(diagnostics, "fnlp candidate redact: {}", error.message());
                if retained {
                    let _ = writeln!(diagnostics, "Durable progress may exist; preserve original inputs and keys, and explicitly authenticate/resume. No rollback is implied.");
                }
                error.exit_code()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(self, input: R, output: W)
        -> Result<(), CandidateError> {
        let (common, limits) = self.validate()?;
        let envelope = self.corpus.envelope(&self, limits)?;
        if self.retention.store_results && !cfg!(all(feature = "asupersync-runtime", feature = "metadata-store",
            target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))) {
            return Err(CandidateError::Unavailable);
        }
        #[cfg(feature = "asupersync-runtime")]
        { runtime::redaction::corpus::execute(self, common, limits, envelope, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, common, limits, envelope, input, output); Err(CandidateError::Unavailable) }
    }
}

#[cfg(test)] mod tests;
