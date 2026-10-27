//! Complete-document exact-schema extraction on one charged INT4 engine.
//! Every chunk identity and full resident KV requirement is checked before
//! native execution. The one invocation retains preparation, reduction and
//! output ownership through final validation, cancellation and physical drain.

use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, grouped_int4::{Int4ExtractionBatchPlanner,
        long::{Int4ExtractionMapError, Int4ExtractionMapLimits, Int4ExtractionMapRun}}},
    native_engine::{constrained::JsonDecodeError, constrained_int4},
    tasks::{extract::{ExtractError, grouped_int4::Int4ExtractError}, ir::TaskBudget},
    validation::grounded_fields::{SourceFieldEvidence, VerifiedSourceSpan},
};

/// Admission for a complete original document. The immutable compiler owns
/// the base identity; this config cannot substitute another model or task.
/// Preparation prices all retained schema/source plans and planner/vocabulary
/// assets. Reduction reserve adds headroom to the explicitly priced frontier,
/// evidence coordinates and token storage. These are modeled ledger charges.
pub struct Int4ExtractionMapConfig {
    pub request: ExtractionBatchArgs,
    pub limits: Int4ExtractionMapLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
}
impl Int4ExtractionMapConfig {
    /// Exact modeled reduction reservation for this admitted chunk count,
    /// including native results, both coordinate copies and retained tokens.
    /// Native, source/planner and final output storage are charged separately.
    pub fn workspace_bytes(&self, chunks: usize) -> Result<u64, HostedError> {
        document_workspace_bytes(self.request.budget, self.limits, chunks, self.reduction_reserve_bytes)
    }
}
struct ExtractionMapInput {
    source: String,
    planner: Arc<Int4ExtractionBatchPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: Int4ExtractionMapConfig,
}

impl NlpEngine {
    /// Produce independently validated schema values for every source chunk.
    /// One model, native engine and control own the entire map/reduce pass.
    /// The final guarded result preserves original coordinates and independent
    /// chunk values; it does not fabricate a merged whole-document object.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int4_document(&self, model: &ResidentInt4, source: String,
        planner: Arc<Int4ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4ExtractionMapConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ExtractionMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        if planner.group_width() != width { return Err(HostedError::ModelIdentity); }
        let identity = planner.base_execution_identity();
        constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
        let required = Int4MemoryRequirement::for_context(config.native.context_tokens, width).map_err(native_error)?;
        validate(&config, identity, width, source.len(), required.kv_bytes)?;
        let bytes = input_bytes(source.capacity(), &config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ExtractionMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Capture the complete storage-before-charge aggregate. Queued
            // cancellation must dispose its owners before releasing the charge.
            let input = input;
            let config = &input.value.config;
            poll(control)?;
            let prepared = input.value.planner.plan_document_with_control(&input.value.source,
                &config.request, config.limits, control).map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let count = expected.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || prepared.execution_identities().len() != count
                || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, config.workspace_bytes(count)?)?;
            let tokens = (count as u64).checked_mul(u64::from(config.request.budget.max_output_tokens))
                .ok_or(HostedError::Limits("INT4 extraction map token arithmetic"))?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count).map_err(|_| map_error(Int4ExtractionMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                poll(control)?;
                constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
                check_identity(identity, width)?;
                let base = input.value.planner.base_execution_identity();
                if identity.tokenizer_digest != base.tokenizer_digest
                    || identity.template_digest != base.template_digest {
                    return Err(HostedError::ModelIdentity);
                }
                admitted.push(identity.clone());
            }
            poll(control)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &mut engine.value,
                    &input.value.vocabulary, control).map_err(map_error)?;
                expected.verify_completed(&result).map_err(map_error)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(map_error(Int4ExtractionMapError::Extraction(ExtractError::InvalidResult.into())));
                }
                poll(control)?;
                Ok(result)
            })?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn map_error(error: Int4ExtractionMapError) -> HostedError { HostedError::Int4(Int4HostError::ExtractionMap(error)) }
fn check_identity(identity: &ExecutionIdentity, width: usize) -> Result<(), HostedError> {
    constrained_int4::check_profile(identity, width).map_err(|_| HostedError::ModelIdentity)?;
    if identity.task_spec != "extract-v1" { return Err(HostedError::ModelIdentity); }
    Ok(())
}
fn validate(config: &Int4ExtractionMapConfig, identity: &ExecutionIdentity, width: usize,
    source_bytes: usize, kv_bytes: u64) -> Result<(), HostedError> {
    check_identity(identity, width)?;
    config.native.run.validate()?;
    config.request.budget.validate().map_err(|_| HostedError::Limits("INT4 extraction map task budget"))?;
    let m = config.limits.mapping; let v = config.limits.verification; let w = m.max_model_work;
    if source_bytes == 0 || source_bytes > m.chunks.max_input_bytes || config.request.schema.is_empty()
        || config.preparation_reserve_bytes == 0 || config.reduction_reserve_bytes == 0
        || config.native.allocator_reserve_bytes == 0 || m.chunks.context_tokens != config.native.context_tokens
        || kv_bytes == 0 || kv_bytes > config.request.budget.max_kv_bytes
        || !(1..=256).contains(&m.chunks.max_chunks)
        || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
        || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
        || w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
        || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0
        || m.reduction.max_result_bytes == 0 || m.reduction.max_live_value_bytes == 0
        || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
        || v.max_scan_steps == 0 {
        return Err(HostedError::Limits("INT4 extraction map source, capacity or whole-document limits"));
    }
    m.chunks.effective_token_limit().map_err(|_| HostedError::Limits("INT4 extraction map chunk geometry"))?;
    Ok(())
}
fn input_bytes(source_capacity: usize, config: &Int4ExtractionMapConfig) -> Result<u64, HostedError> {
    sum(&[source_capacity as u64, config.request.schema.capacity() as u64, config.preparation_reserve_bytes])
}
pub(super) fn document_workspace_bytes(budget: TaskBudget, limits: Int4ExtractionMapLimits,
    chunks: usize, reserve: u64) -> Result<u64, HostedError> {
    if chunks == 0 || chunks > limits.mapping.chunks.max_chunks || chunks > 256 || reserve == 0 {
        return Err(HostedError::Limits("INT4 extraction map partition cardinality or reserve"));
    }
    let native = budget.max_output_bytes.checked_mul(chunks as u64)
        .ok_or(HostedError::Limits("INT4 extraction map native result arithmetic"))?;
    let staged = sum(&[native, limits.mapping.reduction.max_live_value_bytes as u64])?
        .checked_mul(4).ok_or(HostedError::Limits("INT4 extraction map staging arithmetic"))?;
    let tokens = u64::from(budget.max_output_tokens).checked_mul(chunks as u64)
        .and_then(|n| n.checked_mul(8)).ok_or(HostedError::Limits("INT4 extraction map token arithmetic"))?;
    // Price both temporary scan vectors and retained lifted evidence. Each
    // span owns four coordinates; the small occurrence enum is not its size.
    let v = limits.verification;
    let fields = (v.max_fields as u64).checked_mul(std::mem::size_of::<SourceFieldEvidence>() as u64)
        .ok_or(HostedError::Limits("INT4 extraction map field arithmetic"))?;
    let spans = (v.max_matches as u64).checked_mul(std::mem::size_of::<VerifiedSourceSpan>() as u64)
        .and_then(|n| n.checked_mul(2)).ok_or(HostedError::Limits("INT4 extraction map occurrence arithmetic"))?;
    sum(&[staged, tokens, fields, spans, reserve])
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(map_error(Int4ExtractionMapError::Extraction(Int4ExtractError::Extraction(
            ExtractError::Decode(JsonDecodeError::Cancelled(cause)))))),
        None => Ok(()),
    }
}

#[cfg(test)] mod tests;
