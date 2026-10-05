//! Complete-document exact-schema extraction on the process-owned INT8 host.
//! No global object merge, implicit model loading, or per-chunk budget renewal.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, quantized::{Int8ExtractionBatchPlanner,
        long::{Int8ExtractionMapLimits, Int8ExtractionMapRun}}},
    native_engine::{constrained_int8, decode::DecodeStepControl},
    tasks::source_planning::quantized::{Int8SourceError, long::Int8SourceMapError},
    validation::grounded_fields::{SourceFieldEvidence, SourceOccurrence},
};

/// Explicit whole-document admission. The compiler's immutable base identity
/// is authoritative; callers cannot substitute a second identity in this config.
/// Preparation prices ALL retained schema/source plans and immutable compiler
/// assets. Reduction headroom is additional to the payload calculation below.
/// These are modeled ledger reservations, not an allocator or an RSS guarantee.
pub struct ExtractionMapConfig {
    pub request: ExtractionBatchArgs,
    pub limits: Int8ExtractionMapLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
}
struct ExtractionMapInput {
    source: String,
    planner: Arc<Int8ExtractionBatchPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: ExtractionMapConfig,
}

impl NlpEngine {
    /// Extract independent, exact-schema JSON values from every source chunk.
    /// One physical invocation owns all preparation, native work, cancellation
    /// and reduction. The output retains its memory charge after native drain.
    /// Source membership is optional and is never advertised as factual truth.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_document(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: ExtractionMapConfig, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ExtractionMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        let identity = planner.base_execution_identity();
        check_model_identity(model.artifact_identity(), identity)?;
        let required = requirements(config.native)?;
        validate(&config, identity, source.len(), required.kv_bytes)?;
        let bytes = input_bytes(source.capacity(), &config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ExtractionMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Keep the WHOLE owner: disjoint-field capture must never drop the
            // charge while queued source/schema/compiler storage is still live.
            let input = input;
            let config = &input.value.config;
            let prepared = input.value.planner.plan_document_with_control(&input.value.source,
                &config.request, config.limits, control).map_err(HostedError::SourceMap)?;
            let expected = prepared.preflight_metadata();
            let count = expected.chunk_count();
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                reduction_bytes(config, count)?)?;
            let tokens = (count as u64).checked_mul(u64::from(config.request.budget.max_output_tokens))
                .ok_or(HostedError::Limits("extraction map token arithmetic"))?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count)
                .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                poll(control)?;
                check_model_identity(model.artifact_identity(), identity)?;
                check_identity(identity)?;
                // Compilation may bind prompts/schema/policy, never change the
                // tokenizer or template belonging to this immutable compiler.
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
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &mut engine.value,
                    &input.value.vocabulary, control).map_err(HostedError::SourceMap)?;
                expected.verify_completed(&result).map_err(HostedError::SourceMap)?;
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
fn check_identity(identity: &ExecutionIdentity) -> Result<(), HostedError> {
    identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
    if identity.task_spec != "extract-v1" { return Err(HostedError::ModelIdentity); }
    Ok(())
}
fn validate(config: &ExtractionMapConfig, identity: &ExecutionIdentity, source_bytes: usize,
    kv_bytes: u64) -> Result<(), HostedError> {
    check_identity(identity)?;
    config.native.run.validate()?;
    config.request.budget.validate().map_err(|_| HostedError::Limits("extraction map task budget"))?;
    let m = config.limits.mapping;
    let v = config.limits.verification;
    let w = m.max_model_work;
    if source_bytes == 0 || source_bytes > m.chunks.max_input_bytes || config.request.schema.is_empty()
        || config.preparation_reserve_bytes == 0 || config.reduction_reserve_bytes == 0
        || m.chunks.context_tokens != config.native.context_tokens
        || kv_bytes > config.request.budget.max_kv_bytes
        || !(1..=256).contains(&m.chunks.max_chunks)
        || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
        || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
        || w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
        || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0
        || m.reduction.max_result_bytes == 0 || m.reduction.max_live_value_bytes == 0
        || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
        || v.max_scan_steps == 0 {
        return Err(HostedError::Limits("extraction map source, capacity or whole-document limits"));
    }
    m.chunks.effective_token_limit().map_err(|_| HostedError::Limits("extraction map chunk geometry"))?;
    Ok(())
}
fn input_bytes(source_capacity: usize, config: &ExtractionMapConfig) -> Result<u64, HostedError> {
    sum(&[source_capacity as u64, config.request.schema.capacity() as u64, config.preparation_reserve_bytes])
}
fn reduction_bytes(config: &ExtractionMapConfig, chunks: usize) -> Result<u64, HostedError> {
    if chunks == 0 || chunks > config.limits.mapping.chunks.max_chunks || chunks > 256 {
        return Err(HostedError::Limits("extraction map partition cardinality"));
    }
    let b = config.request.budget;
    let native = b.max_output_bytes.checked_mul(chunks as u64)
        .ok_or(HostedError::Limits("extraction map native result arithmetic"))?;
    let staged = sum(&[native, config.limits.mapping.reduction.max_live_value_bytes as u64])?
        .checked_mul(4).ok_or(HostedError::Limits("extraction map staging arithmetic"))?;
    let tokens = u64::from(b.max_output_tokens).checked_mul(chunks as u64)
        .and_then(|n| n.checked_mul(8)).ok_or(HostedError::Limits("extraction map token arithmetic"))?;
    // A coordinate scan may allocate its complete temporary match vector before
    // serialization rejects an oversized result. Price that vector AND the
    // retained original-coordinate evidence, not just the final JSON cap.
    let v = config.limits.verification;
    let fields = (v.max_fields as u64).checked_mul(std::mem::size_of::<SourceFieldEvidence>() as u64)
        .ok_or(HostedError::Limits("extraction map field arithmetic"))?;
    let spans = (v.max_matches as u64).checked_mul(std::mem::size_of::<SourceOccurrence>() as u64)
        .and_then(|n| n.checked_mul(2)).ok_or(HostedError::Limits("extraction map occurrence arithmetic"))?;
    sum(&[staged, tokens, fields, spans, config.reduction_reserve_bytes])
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(HostedError::SourceMap(Int8SourceMapError::Source(Int8SourceError::Cancelled(cause)))),
        None => Ok(()),
    }
}

#[cfg(test)] mod tests;
