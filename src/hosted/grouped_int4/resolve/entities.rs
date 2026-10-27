//! Raw-source NER and resolution share one charged physical invocation.
//! The optional chunked path keeps original document identities and coordinates.
use super::*;
use crate::corpus::entities_int4::{Int4EntityConfig, Int4EntityError, Int4EntityRun,
    PreparedInt4EntityCorpus, INT4_ENTITY_EXECUTION};

struct Input { prepared: Option<PreparedInt4EntityCorpus>, vocabulary: Arc<ExtractionVocabulary> }

impl NlpEngine {
    /// Execute a consumed, model-bound raw-document corpus. Every NER chunk
    /// was preflighted and is rebuilt against its private witness; one native
    /// engine then scores the resulting complete graph. Preparation covers
    /// pinned planners/configuration/allocator overhead; graph reserve covers
    /// expanded mentions, all pair plans, receipts and clustering. These are
    /// explicit modeled commitments, not measurements of process RSS.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_entities(&self, model: &ResidentInt4, prepared: PreparedInt4EntityCorpus,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits, preparation_reserve_bytes: u64,
        graph_reserve_bytes: u64, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4EntityRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(prepared.source_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        constrained_int4::check_model(prepared.resolution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        if prepared.group_width() != width { return Err(HostedError::ModelIdentity); }
        let (required, scratch_bytes) = requirements(prepared.config(), prepared.required_ner_context_tokens(),
            native, preparation_reserve_bytes, graph_reserve_bytes, width)?;
        let input_bytes = sum(&[prepared.retained_input_bytes().map_err(execution_error)?, preparation_reserve_bytes])?;
        let temporary_bytes = temporary_bytes(prepared.config(), graph_reserve_bytes)?;
        let output_bytes = prepared.config().max_result_bytes as u64;
        let expected = Expected { width, documents: prepared.document_count(), chunks: prepared.chunk_count(),
            ner_work: prepared.ner_reserved_work(), masks: prepared.reserved_mask_visits(),
            max_model_work: prepared.config().max_model_work, chunked: prepared.config().chunks.is_some() };
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(Input { prepared: Some(prepared), vocabulary }))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            // Moving the consumed plan must not release the charge that owns
            // its source/witness storage while NER and graph stages still run.
            let mut input = input;
            let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
            let output = output_claim(&lease, output_bytes, 0)?;
            let result = if expected.documents == 0 {
                let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
                allocate(output, || {
                    let result = prepared.finalize_without_model(control).map_err(execution_error)?;
                    check_result(&result, expected)?;
                    checkpoint(control)?;
                    Ok(result)
                })?
            } else {
                let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
                let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(&model.inner.loaded.value, native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound,
                    }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
                })?;
                let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
                let result = allocate(output, || {
                    let result = prepared.execute_with_control(&mut engine.value, &input.value.vocabulary, control)
                        .map_err(execution_error)?;
                    check_result(&result, expected)?;
                    if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                        return Err(execution_error(Int4EntityError::Accounting));
                    }
                    checkpoint(control)?;
                    Ok(result)
                })?;
                drop(engine);
                result
            };
            drop(temporary);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

pub(super) fn requirements(config: &Int4EntityConfig, ner_context: usize, native: NativeLimits,
    preparation: u64, graph: u64, width: usize) -> Result<(Int4MemoryRequirement, u64), HostedError> {
    native.run.validate()?;
    config.validate().map_err(execution_error)?;
    let required = Int4MemoryRequirement::for_context(native.context_tokens, width).map_err(native_error)?;
    if preparation == 0 || graph == 0 || native.allocator_reserve_bytes == 0
        || ner_context > native.context_tokens || config.source_planning.max_context_tokens > native.context_tokens
        || config.scoring.planning.max_context_tokens > native.context_tokens
        || required.kv_bytes > config.ner_budget.max_kv_bytes
        || required.kv_bytes > config.scoring.planning.per_head.max_kv_bytes {
        return Err(HostedError::Limits("INT4 entity preparation, graph or full resident context"));
    }
    let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes])?;
    Ok((required, scratch))
}

pub(super) fn temporary_bytes(config: &Int4EntityConfig, graph: u64) -> Result<u64, HostedError> {
    // At most one source grammar/native result is resident at a time. Graph
    // storage separately owns the expanded mentions and compact chunk receipts.
    let result = config.ner_budget.max_output_bytes.checked_mul(4)
        .ok_or(HostedError::Limits("INT4 entity NER intermediate arithmetic"))?;
    let tokens = u64::from(config.ner_budget.max_output_tokens).checked_mul(8)
        .ok_or(HostedError::Limits("INT4 entity token arithmetic"))?;
    sum(&[result, tokens, graph])
}

#[derive(Clone, Copy)]
pub(super) struct Expected {
    pub(super) width: usize, pub(super) documents: usize, pub(super) chunks: usize,
    pub(super) ner_work: Int4Work, pub(super) masks: u64,
    pub(super) max_model_work: Int4Work, pub(super) chunked: bool,
}

pub(super) fn check_result(result: &Int4EntityRun, expected: Expected) -> Result<(), HostedError> {
    let resolution = &result.resolution;
    let heads = resolution.result.judgments.len().checked_mul(2)
        .ok_or_else(|| execution_error(Int4EntityError::Accounting))?;
    super::check_result(resolution, expected.width, heads, resolution.planned_model_work)?;
    let reserved = expected.ner_work.checked_add(resolution.planned_model_work).map_err(native_error)?;
    let actual = result.ner_model_work.checked_add(resolution.model_work).map_err(native_error)?;
    let chunk_profile = expected.chunked.then_some(crate::tasks::mapreduce::CHUNK_PROFILE);
    if result.schema_version != 1 || result.execution != INT4_ENTITY_EXECUTION
        || result.numerics_profile != STRICT_INT4_PROFILE || result.group_width != expected.width
        || result.extraction_task != crate::tasks::ner::NER_TASK_VERSION || result.chunk_profile != chunk_profile
        || result.documents.len() != expected.documents || resolution.result.document_count != expected.documents
        || result.chunk_count != expected.chunks || result.ner_reserved_work != expected.ner_work
        || result.reserved_mask_node_visits != expected.masks || result.mask_node_visit_charge > expected.masks
        || result.reserved_model_work != reserved || result.model_work != actual {
        return Err(execution_error(Int4EntityError::Accounting));
    }
    super::check_work(result.ner_model_work, expected.ner_work)?;
    super::check_work(actual, reserved)?;
    super::check_work(reserved, expected.max_model_work)?;
    Ok(())
}

pub(super) fn execution_error(error: Int4EntityError) -> HostedError { HostedError::Int4(Int4HostError::Entities(error)) }

#[cfg(test)]
mod tests;
