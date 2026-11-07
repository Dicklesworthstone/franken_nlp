//! Pure ownership quote for ordered complete-document question answering.
//! Shared by CLI and host before private population IO. A quote describes
//! modeled simultaneous ownership; it is not a permit or measured RSS bound.
use super::{Int4QuestionDocumentBatchConfig, QUESTION_DOCUMENT_CHUNK_METADATA_BYTES};
use crate::{
    batch::BatchLimits,
    native_engine::strict_int4::Int4MemoryRequirement,
    tasks::{answer::{PassageCitation, PassageOccurrence}, summarize::SourceCitation,
        source_planning::grouped_int4::long::{Int4SourceMapError,
            cohort::{Int4SourceMapCohortMemoryLimits, Int4SourceMapCohortRequirement,
                COHORT_ENVELOPE_BYTES}}},
    validation::grounded_fields::{SourceFieldEvidence, VerifiedSourceSpan},
};

/// Compiler, tokenizer, vocabulary and allocation slack remain explicit host
/// declarations. Complete source/question payloads, frozen identities, native
/// peaks, evidence and result staging are additionally priced by this module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4QuestionDocumentMemoryLimits {
    pub context_tokens: usize,
    pub allocator_reserve_bytes: u64,
    pub preparation_reserve_bytes: u64,
    pub io_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
    /// Explicit discovery schedule; synthesis remains a later scalar phase.
    pub cohort_chunks: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4QuestionDocumentMemoryRequirement {
    pub native: Int4MemoryRequirement,
    /// The complete native peak plus declared allocator slack.
    pub workspace_bytes: u64,
    /// Transport, source/question representations, compiler/IO declarations
    /// and every possible frozen/admitted chunk and continuation identity.
    pub input_bytes: u64,
    /// Entire map frontier, both evidence pools, temporary native transcripts,
    /// optional evidence collection and native discovery decoder storage.
    pub document_workspace_bytes: u64,
    /// Semantic output, canonical serialization and fixed output overhead.
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
    /// Independent and synthesized QA keep text/citations, not token-id arrays.
    /// Native transcripts are temporary and priced in document workspace.
    pub max_output_tokens: u64,
}

impl Int4QuestionDocumentMemoryLimits {
    pub fn requirements(self, width: usize, config: Int4QuestionDocumentBatchConfig,
        transport: BatchLimits) -> Result<Int4QuestionDocumentMemoryRequirement, Int4SourceMapError> {
        config.validate().map_err(|_| Int4SourceMapError::InvalidLimits)?;
        super::check_synthesis_reserve(&config, width).map_err(|_| Int4SourceMapError::InvalidLimits)?;
        transport.validate().map_err(|_| Int4SourceMapError::InvalidLimits)?;
        if self.allocator_reserve_bytes == 0 || self.preparation_reserve_bytes == 0
            || self.io_reserve_bytes == 0 || self.reduction_reserve_bytes == 0
            || self.context_tokens != config.planning.max_context_tokens
            || self.context_tokens != config.mapping.chunks.context_tokens
            || config.mapping.reduction.max_result_bytes > transport.max_output_line_bytes {
            return Err(Int4SourceMapError::InvalidLimits);
        }
        let row = Int4MemoryRequirement::for_context(self.context_tokens, width)
            .map_err(|_| Int4SourceMapError::InvalidLimits)?;
        // A wider discovery window cannot expand any individual task's KV.
        if row.kv_bytes != config.budget.max_kv_bytes { return Err(Int4SourceMapError::Admission); }
        let cohort = self.cohort_requirements(width, config)?;
        let native = cohort.map_or(row, |quote| quote.native);
        let chunks = config.mapping.chunks.max_chunks as u64;
        let synthesis_passes = u64::from(config.synthesis.is_some());
        let identities = sum(&[chunks, synthesis_passes])?;
        // The population record, all simultaneous chunk plans, and an optional
        // continuation retain their own question representations. The 64-byte
        // modeled per-byte allowance covers copied text, encoded token/context
        // rails and canonical staging; compiler graphs remain separately priced.
        let question_copies = sum(&[chunks, 1, synthesis_passes])?;
        let question_payload = mul(mul(config.max_question_bytes as u64, question_copies)?, 64)?;
        let source_payload = mul(config.mapping.chunks.max_input_bytes as u64, 64)?;
        let transport_bytes = sum(&[mul(transport.max_line_bytes as u64, 8)?,
            mul(transport.max_output_line_bytes as u64, 4)?, mul(transport.max_epoch_ids as u64, 128)?,
            transport.max_epoch_id_bytes as u64, self.preparation_reserve_bytes, self.io_reserve_bytes])?;
        let input_bytes = sum(&[transport_bytes, source_payload, question_payload,
            mul(identities, QUESTION_DOCUMENT_CHUNK_METADATA_BYTES)?])?;

        let budget = config.budget; let mapping = config.mapping;
        // Whitespace source ranges still occupy the complete ordered frontier.
        // Price max_chunks, never only the number of native discovery rows.
        let staged = mul(sum(&[mul(budget.max_output_bytes, chunks)?,
            mapping.reduction.max_live_value_bytes as u64])?, 4)?;
        let tokens = mul(u64::from(budget.max_output_tokens), chunks)?;
        let map_fields = mul(mapping.verification.max_fields as u64,
            std::mem::size_of::<SourceFieldEvidence>() as u64)?;
        let map_spans = mul(mapping.verification.max_matches as u64,
            mul(std::mem::size_of::<VerifiedSourceSpan>() as u64, 2)?)?;
        // Native passage-local citations, the independent verifier, and lifted
        // original-coordinate citations can coexist. The two verification
        // pools are distinct; neither substitutes for the other's allowance.
        let qa_fields = mul(config.verification.max_fields as u64, sum(&[
            std::mem::size_of::<PassageCitation>() as u64,
            std::mem::size_of::<SourceCitation>() as u64])?)?;
        let qa_spans = mul(config.verification.max_matches as u64, sum(&[
            std::mem::size_of::<PassageOccurrence>() as u64,
            mul(std::mem::size_of::<VerifiedSourceSpan>() as u64, 2)?])?)?;
        let synthesis = synthesis_workspace(config)?;
        let cohort_workspace = match cohort {
            Some(quote) => sum(&[quote.workspace_bytes, mul(COHORT_ENVELOPE_BYTES, 4)?])?,
            None => 0,
        };
        let document_workspace_bytes = sum(&[staged, mul(tokens, 8)?, map_fields, map_spans,
            qa_fields, qa_spans, self.reduction_reserve_bytes, synthesis, cohort_workspace])?;
        let max_output_bytes = mapping.reduction.max_result_bytes as u64;
        let required = Int4QuestionDocumentMemoryRequirement {
            native, workspace_bytes: sum(&[native.rope_bytes, native.scratch_payload_bound,
                self.allocator_reserve_bytes])?, input_bytes, document_workspace_bytes,
            output_reservation_bytes: sum(&[mul(max_output_bytes, 4)?, 4096])?,
            max_output_bytes, max_output_tokens: 0,
        };
        // Each component may fit u64 while their simultaneous ownership does
        // not. Reject before callers can construct a wrapped total admission.
        sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
            required.document_workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }

    pub(crate) fn cohort_requirements(self, width: usize, config: Int4QuestionDocumentBatchConfig)
        -> Result<Option<Int4SourceMapCohortRequirement>, Int4SourceMapError> {
        self.cohort_chunks.map(|chunks| {
            if chunks != config.mapping.reduction.map_batch_chunks || chunks > config.mapping.chunks.max_chunks {
                return Err(Int4SourceMapError::InvalidLimits);
            }
            let quote = Int4SourceMapCohortMemoryLimits { max_chunks: chunks,
                context_tokens: self.context_tokens,
                max_output_tokens_per_chunk: u64::from(config.budget.max_output_tokens),
                max_result_bytes_per_chunk: config.budget.max_output_bytes }.requirements(width)?;
            if config.synthesis.is_some() {
                // Every discovery window drains before evidence synthesis.
                // Use componentwise maxima, never sum two simultaneous engines.
                quote.with_synthesis_context(self.context_tokens, width, config.budget.max_kv_bytes)
            } else { Ok(quote) }
        }).transpose()
    }
}

fn synthesis_workspace(config: Int4QuestionDocumentBatchConfig) -> Result<u64, Int4SourceMapError> {
    let Some(limits) = config.synthesis else { return Ok(0); };
    // All verified quotes survive collection; no top-k truncation. Verbatim
    // copies, joined/encoded evidence, passage manifest/layout, origin fanout,
    // independent scans and the final native transcript coexist with discovery.
    sum(&[mul(limits.max_evidence_bytes as u64, 64)?,
        mul(limits.max_evidence_passages as u64, 1024)?,
        mul(config.verification.max_matches as u64, 128)?,
        mul(config.verification.max_fields as u64, 128)?,
        mul(config.budget.max_output_bytes, 4)?,
        mul(u64::from(config.budget.max_output_tokens), 8)?])
}
fn sum(values: &[u64]) -> Result<u64, Int4SourceMapError> {
    values.iter().try_fold(0_u64, |n, value| n.checked_add(*value).ok_or(Int4SourceMapError::InvalidLimits))
}
fn mul(left: u64, right: u64) -> Result<u64, Int4SourceMapError> {
    left.checked_mul(right).ok_or(Int4SourceMapError::InvalidLimits)
}

#[cfg(test)] mod tests;
