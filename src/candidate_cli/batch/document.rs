//! Opt-in complete-document extraction on the existing corpus CLI.
//! One event per document, not one event or renewed allowance per chunk.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, quantized::long::{Int8ExtractionMapLimits,
        stream::ExtractionDocumentBatchConfig}},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::quantized::long::Int8SourceMapLimits},
    validation::grounded_fields::GroundingBudget,
};
const MAX_DOCUMENT_RESULT: usize = 64 * 1024 * 1024;

#[derive(Args, Default)]
pub(in crate::candidate_cli) struct DocumentArgs {
    /// Extract only: partition each complete document and retain independent
    /// exact-schema JSON values. Requires a fixed shared --schema, not task_args.
    #[arg(long, requires = "schema", conflicts_with = "defaults")]
    pub chunked: bool,
    /// Maximum chunks per original document (default 64); never truncate.
    #[arg(long, requires = "chunked")]
    max_chunks: Option<usize>,
    /// Additional chunk byte ceiling; actual schema/context fit is rechecked.
    #[arg(long, requires = "chunked")]
    max_chunk_bytes: Option<usize>,
    /// Complete per-document result cap, excluding transport framing (default 16777216).
    #[arg(long, requires = "chunked")]
    max_document_result_bytes: Option<usize>,
    /// Additional reduction/allocator headroom in MiB (default 64), not measured RSS.
    #[arg(long, requires = "chunked")]
    reduction_reserve_mib: Option<u64>,
    /// Whole-document independent verification fields (default 4096).
    #[arg(long, requires = "chunked")]
    max_extraction_fields: Option<usize>,
    /// Whole-document original-coordinate occurrence spans (default 16384).
    #[arg(long, requires = "chunked")]
    max_extraction_evidence_spans: Option<usize>,
    /// Whole-document independent verification scan steps (default 67108864).
    #[arg(long, requires = "chunked")]
    max_extraction_scan_steps: Option<u64>,
}
struct Settings { chunks: usize, chunk_bytes: usize, result_bytes: usize, reduction_bytes: u64, verification: GroundingBudget }
impl BatchCommand {
    fn document_settings(&self) -> Result<Option<Settings>, CandidateError> {
        let d = &self.document;
        if !d.chunked {
            if d.max_chunks.is_some() || d.max_chunk_bytes.is_some() || d.max_document_result_bytes.is_some()
                || d.reduction_reserve_mib.is_some() || d.max_extraction_fields.is_some()
                || d.max_extraction_evidence_spans.is_some() || d.max_extraction_scan_steps.is_some() {
                return Err(CandidateError::Arguments);
            }
            return Ok(None);
        }
        self.validate_extraction_flags()?;
        if self.task != "extract" || self.schema.is_none() || self.defaults.is_some()
            || self.host.max_input_bytes < 4 { return Err(CandidateError::Arguments); }
        let defaults = GroundingBudget::default();
        let settings = Settings { chunks: d.max_chunks.unwrap_or(64),
            chunk_bytes: d.max_chunk_bytes.unwrap_or(self.host.context_tokens).min(self.host.max_input_bytes),
            result_bytes: d.max_document_result_bytes.unwrap_or(16 * 1024 * 1024),
            reduction_bytes: d.reduction_reserve_mib.unwrap_or(64).checked_mul(MIB).ok_or(CandidateError::Arguments)?,
            verification: GroundingBudget { max_fields: d.max_extraction_fields.unwrap_or(defaults.max_fields),
                max_matches: d.max_extraction_evidence_spans.unwrap_or(defaults.max_matches),
                max_scan_steps: d.max_extraction_scan_steps.unwrap_or(defaults.max_scan_steps) } };
        let v = settings.verification;
        if !(1..=256).contains(&settings.chunks) || settings.chunk_bytes < 4
            || d.max_chunk_bytes.is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || !(1..=MAX_DOCUMENT_RESULT).contains(&settings.result_bytes)
            || settings.result_bytes < self.host.max_result_bytes || settings.reduction_bytes == 0
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || !(1..=1_000_000_000_000).contains(&v.max_scan_steps) { return Err(CandidateError::Arguments); }
        Ok(Some(settings))
    }
    pub(super) fn document_envelope(&self, limits: Limits, mut envelope: CorpusEnvelope)
        -> Result<CorpusEnvelope, CandidateError> {
        let Some(d) = self.document_settings()? else { return Ok(envelope); };
        // Every retained prompt/schema/grammar is priced, not just one context.
        // The host separately charges native outputs and the live map frontier.
        let per_chunk = (self.host.context_tokens as u64).checked_mul(32)
            .and_then(|n| n.checked_add(u64::from(self.host.max_grammar_states).checked_mul(512)?))
            .and_then(|n| n.checked_add(2 * crate::candidate_cli::extract::SCHEMA_BYTES as u64))
            .ok_or(CandidateError::Arguments)?;
        let floor = per_chunk.checked_mul(d.chunks as u64)
            .and_then(|n| n.checked_add(256 * MIB))
            .and_then(|n| n.checked_add((self.host.max_input_bytes as u64).checked_mul(64)?))
            .and_then(|n| n.checked_add((d.result_bytes as u64 + 4096).checked_mul(2)?))
            .ok_or(CandidateError::Arguments)?;
        if limits.preparation_bytes < floor || d.reduction_bytes > limits.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        // The incoming envelope already prices max_requests independent
        // contexts. Scale by maximum chunks, NOT one huge attention triangle.
        envelope.native.max_model_work = scale_work(envelope.native.max_model_work, d.chunks as u64)?;
        envelope.native.masks.max_visits_per_item = envelope.native.masks.max_visits_per_item
            .checked_mul(d.chunks as u64).ok_or(CandidateError::Arguments)?;
        envelope.native.masks.max_visits_per_run = envelope.native.masks.max_visits_per_run
            .checked_mul(d.chunks as u64).ok_or(CandidateError::Arguments)?;
        let work = envelope.native.max_model_work;
        envelope.transport.max_work = BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits };
        envelope.transport.max_output_line_bytes = d.result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?;
        // Still one result per document. N+4 candidate framing remains exactly
        // the original corpus allowance; chunk count creates no extra events.
        envelope.io_bytes = (envelope.transport.max_output_line_bytes as u64).checked_add(FRAME_ALLOWANCE)
            .and_then(|n| n.checked_add(IO_BUFFER_BYTES as u64 * 2)).ok_or(CandidateError::Arguments)?;
        envelope.transport.validate().map_err(|_| CandidateError::Arguments)?;
        crate::batch::source::quantized::validate_limits(envelope.native).map_err(|_| CandidateError::Arguments)?;
        Ok(envelope)
    }
    pub(in crate::candidate_cli) fn document_configuration(&self, request: ExtractionBatchArgs,
        envelope: CorpusEnvelope) -> Result<ExtractionDocumentBatchConfig, CandidateError> {
        let d = self.document_settings()?.ok_or(CandidateError::Arguments)?;
        let one = constrained_int8::planned_work(self.host.context_tokens - self.host.max_new_tokens,
            self.host.max_new_tokens).map_err(|_| CandidateError::Arguments)?;
        let per_document = scale_work(one, d.chunks as u64)?;
        let masks = self.host.max_mask_node_visits.checked_mul(d.chunks as u64).ok_or(CandidateError::Arguments)?;
        let count = usize::try_from(self.max_requests).map_err(|_| CandidateError::Arguments)?;
        let verification = GroundingBudget {
            max_fields: d.verification.max_fields.checked_mul(count).ok_or(CandidateError::Arguments)?,
            max_matches: d.verification.max_matches.checked_mul(count).ok_or(CandidateError::Arguments)?,
            max_scan_steps: d.verification.max_scan_steps.checked_mul(self.max_requests).ok_or(CandidateError::Arguments)?,
        };
        let config = ExtractionDocumentBatchConfig { request,
            limits: Int8ExtractionMapLimits { mapping: Int8SourceMapLimits {
                chunks: ChunkLimits { max_input_bytes: self.host.max_input_bytes, max_chunk_bytes: d.chunk_bytes,
                    max_chunk_tokens: self.host.context_tokens, context_tokens: self.host.context_tokens,
                    reserved_tokens: self.host.max_new_tokens, max_chunks: d.chunks, max_tokenizer_calls: 8192 },
                reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 8, max_reduction_levels: 8, max_task_calls: 1024,
                    max_value_bytes: d.result_bytes, max_live_value_bytes: MAX_DOCUMENT_RESULT,
                    max_total_value_bytes: MAX_DOCUMENT_RESULT, max_result_bytes: d.result_bytes },
                max_model_work: per_document, mask_limits: self.host.masks(), mask_visits_per_chunk: self.host.max_mask_node_visits,
                max_mask_visits: masks }, verification: d.verification },
            max_model_work: envelope.native.max_model_work,
            max_mask_visits: envelope.native.masks.max_visits_per_run, max_verification: verification };
        config.validate().map_err(|_| CandidateError::Arguments)?;
        Ok(config)
    }
    pub(in crate::candidate_cli) fn document_reduction_bytes(&self) -> Result<u64, CandidateError> {
        Ok(self.document_settings()?.ok_or(CandidateError::Arguments)?.reduction_bytes)
    }
}

#[cfg(test)] mod tests;
