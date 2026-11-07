//! The same complete-document admission contract serves live and durable sinks.
use super::*;
use crate::{grammar::mask::MaskWorkLimits, native_engine::strict_int4::Int4Work};

pub(in crate::hosted::grouped_int4) struct DocumentAdmission<'a> {
    pub(in crate::hosted::grouped_int4) lease: &'a EngineLease,
    pub(in crate::hosted::grouped_int4) model: &'a Int4ModelIdentity,
    pub(in crate::hosted::grouped_int4) task_spec: &'a str,
    pub(in crate::hosted::grouped_int4) kv_bytes: u64,
    pub(in crate::hosted::grouped_int4) mapping: Int4SourceMapLimits,
    pub(in crate::hosted::grouped_int4) result_bytes: u64,
    pub(in crate::hosted::grouped_int4) output_tokens: u64,
}
impl Int4SourceDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4SourceDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        check_document_request(self.model.group_width(), self.task_spec, self.kv_bytes,
            self.mapping, self.result_bytes, self.output_tokens, &request)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        for identity in request.identities {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            identities.push(identity.clone());
        }
        let guard = output_claim(self.lease, request.max_result_bytes, request.max_output_tokens)
            .map_err(|error| match error {
                HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
                _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
            })?;
        Ok((identities, guard))
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::hosted::grouped_int4) fn check_document_request(width: usize, task_spec: &str, kv_bytes: u64,
    mapping: Int4SourceMapLimits, result_bytes: u64, output_tokens: u64,
    request: &Int4SourceDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    if !matches!(task_spec, "ner-v1" | "keyphrases-v1" | "summarize-v1" | "answer-v1")
        || request.chunk_count == 0 || request.chunk_count > mapping.chunks.max_chunks
        || request.chunk_count > 256 || request.identities.is_empty()
        || request.identities.len() > request.chunk_count
        || request.kv_reservation_bytes != kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    for identity in request.identities {
        constrained_int4::check_profile(identity, width)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != task_spec { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    }
    if request.mask_node_visits == 0 || request.mask_node_visits > mapping.max_mask_visits
        || !same_masks(request.mask_limits, mapping.mask_limits) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let work = request.model_work;
    if work.forward_positions == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || work.projected_logits == 0 || work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || work.projected_logits / NANBEIGE_VOCAB_SIZE as u64 > work.forward_positions
        || !work_fits(work, mapping.max_model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    // This is the sum of real independent sequences and reserved future passes,
    // not a fabricated context-sized sequence or the INT8 projection formula.
    if request.max_result_bytes == 0 || request.max_result_bytes != result_bytes
        || request.max_output_tokens > output_tokens {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}

fn same_masks(left: MaskWorkLimits, right: MaskWorkLimits) -> bool {
    left.max_trie_node_visits == right.max_trie_node_visits
        && left.checkpoint_interval_nodes == right.checkpoint_interval_nodes
}
pub(in crate::hosted::grouped_int4) fn work_fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs
        && work.projections.dot_products <= ceiling.projections.dot_products
        && work.projections.multiply_accumulates <= ceiling.projections.multiply_accumulates
}
