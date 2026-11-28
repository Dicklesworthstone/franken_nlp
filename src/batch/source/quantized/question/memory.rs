//! Pure simultaneous-ownership quote for complete-document INT8 QA streams.
//! Shared by CLI and host before private IO; this grants no execution permit
//! and describes modeled payloads plus explicit caller-priced overhead, not RSS.
use super::{Int8QuestionDocumentBatchConfig, QUESTION_DOCUMENT_CHUNK_METADATA_BYTES};
use crate::{
    batch::BatchLimits,
    native_engine::strict_int8::Int8MemoryRequirement,
    tasks::{answer::{PassageCitation, PassageOccurrence}, summarize::SourceCitation,
        source_planning::quantized::long::Int8SourceMapError},
    validation::grounded_fields::{SourceFieldEvidence, VerifiedSourceSpan},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8QuestionDocumentMemoryLimits {
    pub context_tokens: usize,
    pub allocator_reserve_bytes: u64,
    pub preparation_reserve_bytes: u64,
    pub io_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8QuestionDocumentMemoryRequirement {
    pub native: Int8MemoryRequirement,
    /// Native RoPE/scratch and the explicitly declared allocator allowance.
    pub workspace_bytes: u64,
    /// Ordinary corpus transport/preparation/IO plus all complete source,
    /// private question and frozen/reconstructed/admitted identity owners.
    pub input_bytes: u64,
    /// Complete map frontier, native transcripts, citation verification and
    /// lifted original-coordinate evidence; whitespace chunks still count.
    pub document_workspace_bytes: u64,
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
    /// Independent QA retains semantic answers/citations, not token arrays.
    pub max_output_tokens: u64,
}
impl Int8QuestionDocumentMemoryLimits {
    pub fn requirements(self, config: Int8QuestionDocumentBatchConfig, transport: BatchLimits)
        -> Result<Int8QuestionDocumentMemoryRequirement, Int8SourceMapError> {
        config.validate().map_err(|_| Int8SourceMapError::InvalidLimits)?;
        transport.validate().map_err(|_| Int8SourceMapError::InvalidLimits)?;
        if self.allocator_reserve_bytes == 0 || self.preparation_reserve_bytes == 0
            || self.io_reserve_bytes == 0 || self.reduction_reserve_bytes == 0
            || self.context_tokens != config.planning.max_context_tokens
            || self.context_tokens != config.mapping.chunks.context_tokens
            || config.mapping.chunks.max_input_bytes > transport.max_document_bytes
            || config.mapping.reduction.max_result_bytes > transport.max_output_line_bytes {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        let native = Int8MemoryRequirement::for_context(self.context_tokens)
            .map_err(|_| Int8SourceMapError::InvalidLimits)?;
        if native.kv_bytes != config.budget.max_kv_bytes { return Err(Int8SourceMapError::Admission); }
        let chunks = config.mapping.chunks.max_chunks as u64;
        // Original record plus every simultaneous passage plan retains its
        // question. The per-byte bound covers text/encoded/context/canonical
        // rails; source compiler graphs and IO remain explicitly priced above.
        let question_payload = mul(mul(config.max_question_bytes as u64, sum(&[chunks, 1])?)?, 64)?;
        let source_payload = mul(config.mapping.chunks.max_input_bytes as u64, 64)?;
        // Exactly the common CorpusLimits transport quote, permitting the host
        // to add the document delta without double-counting its existing guard.
        let transport_bytes = sum(&[mul(transport.max_line_bytes as u64, 8)?,
            mul(transport.max_output_line_bytes as u64, 4)?, mul(transport.max_epoch_ids as u64, 128)?,
            transport.max_epoch_id_bytes as u64, self.preparation_reserve_bytes, self.io_reserve_bytes])?;
        let input_bytes = sum(&[transport_bytes, source_payload, question_payload,
            mul(chunks, QUESTION_DOCUMENT_CHUNK_METADATA_BYTES)?])?;

        let staged = mul(sum(&[mul(config.budget.max_output_bytes, chunks)?,
            config.mapping.reduction.max_live_value_bytes as u64])?, 4)?;
        let tokens = mul(u64::from(config.budget.max_output_tokens), chunks)?;
        // One native source verifier can coexist with the complete QA lift
        // pool. The latter has its own whole-document/run scan authority.
        let source = config.planning.source.verification;
        let source_fields = mul(source.max_fields as u64, std::mem::size_of::<SourceFieldEvidence>() as u64)?;
        let source_spans = mul(source.max_matches as u64, mul(std::mem::size_of::<VerifiedSourceSpan>() as u64, 2)?)?;
        let qa_fields = mul(config.verification.max_fields as u64, sum(&[
            std::mem::size_of::<PassageCitation>() as u64, std::mem::size_of::<SourceCitation>() as u64])?)?;
        let qa_spans = mul(config.verification.max_matches as u64, sum(&[
            std::mem::size_of::<PassageOccurrence>() as u64, mul(std::mem::size_of::<VerifiedSourceSpan>() as u64, 2)?])?)?;
        let document_workspace_bytes = sum(&[staged, mul(tokens, 8)?, source_fields, source_spans,
            qa_fields, qa_spans, self.reduction_reserve_bytes])?;
        let max_output_bytes = config.mapping.reduction.max_result_bytes as u64;
        let required = Int8QuestionDocumentMemoryRequirement {
            native, workspace_bytes: sum(&[native.rope_bytes, native.scratch_payload_bound, self.allocator_reserve_bytes])?,
            input_bytes, document_workspace_bytes,
            output_reservation_bytes: sum(&[mul(max_output_bytes, 4)?, 4096])?,
            max_output_bytes, max_output_tokens: 0,
        };
        sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
            required.document_workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }
}
fn sum(values: &[u64]) -> Result<u64, Int8SourceMapError> {
    values.iter().try_fold(0_u64, |n, value| n.checked_add(*value).ok_or(Int8SourceMapError::InvalidLimits))
}
fn mul(left: u64, right: u64) -> Result<u64, Int8SourceMapError> {
    left.checked_mul(right).ok_or(Int8SourceMapError::InvalidLimits)
}

#[cfg(test)] mod tests;
