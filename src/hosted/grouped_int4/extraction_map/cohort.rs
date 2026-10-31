//! Whole-document admission with exact native cohorts for successive map windows.
use super::*;
use crate::batch::extract::grouped_int4::long::cohort::{
    Int4ExtractionMapCohortBudget, Int4ExtractionMapCohortMemoryLimits,
    Int4ExtractionMapCohortRequirement, Int4ExtractionMapCohortRun,
};

impl Int4ExtractionMapConfig {
    /// Model-free upper bound for one simultaneous map window. Document
    /// preparation, reduction, output and allocator allowances are separate.
    /// Per-chunk task budgets remain unchanged by the scheduling width.
    pub fn cohort_requirements(&self, group_width: usize)
        -> Result<Int4ExtractionMapCohortRequirement, HostedError> {
        let chunks = self.limits.mapping.reduction.map_batch_chunks;
        if chunks > self.limits.mapping.chunks.max_chunks {
            return Err(HostedError::Limits("INT4 extraction cohort exceeds document chunk ceiling"));
        }
        Int4ExtractionMapCohortMemoryLimits {
            max_chunks: chunks,
            context_tokens: self.native.context_tokens,
            max_output_tokens_per_chunk: u64::from(self.request.budget.max_output_tokens),
            max_result_bytes_per_chunk: self.request.budget.max_output_bytes,
        }.requirements(group_width).map_err(map_error)
    }
}

impl NlpEngine {
    /// Execute each bounded map window through shared native decoder layers.
    /// All real chunk plans, identities and work allowances are checked before
    /// the first forward. Independent JSON values and original source evidence
    /// retain the serial document semantics; the outer stats describe grouping.
    ///
    /// One invocation owns the complete document. Exact-size engines, including
    /// the final short window, live inside a retained peak reservation. A later
    /// chunk, reduction or delivery failure cannot publish a successful prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int4_document_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<Int4ExtractionBatchPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4ExtractionMapConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ExtractionMapCohortRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        if planner.group_width() != width { return Err(HostedError::ModelIdentity); }
        let identity = planner.base_execution_identity();
        constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
        // Each task authorizes one context, never the sum of its neighbors' KV.
        let row = Int4MemoryRequirement::for_context(config.native.context_tokens, width).map_err(native_error)?;
        validate(&config, identity, width, source.len(), row.kv_bytes)?;
        let ceiling = config.cohort_requirements(width)?;
        let bytes = input_bytes(source.capacity(), &config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ExtractionMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            poll(control)?;
            let prepared = input.value.planner.plan_document_with_control(&input.value.source,
                &config.request, config.limits, control).map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let count = expected.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || prepared.execution_identities().len() != count || row.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            // Every window, including the final one, must fit before allocating
            // any native engine. Separate component peaks conservatively cover
            // ragged contexts without inventing padded causal positions.
            let required = prepared.cohort_requirements(config.native.context_tokens).map_err(map_error)?;
            check_peak(required, ceiling)?;
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, config.workspace_bytes(count)?)?;
            let tokens = (count as u64).checked_mul(u64::from(config.request.budget.max_output_tokens))
                .ok_or(HostedError::Limits("INT4 extraction map cohort token arithmetic"))?;
            let output_bytes = sum(&[prepared.max_result_bytes(), 4096])?;
            let output = output_claim(&lease, output_bytes, tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count).map_err(|_| map_error(Int4ExtractionMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                poll(control)?;
                constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
                check_identity(identity, width)?;
                let base = input.value.planner.base_execution_identity();
                if identity.tokenizer_digest != base.tokenizer_digest || identity.template_digest != base.template_digest {
                    return Err(HostedError::ModelIdentity);
                }
                admitted.push(identity.clone());
            }
            poll(control)?;
            // Live reservations cover changing exact-sized engines. They are
            // not committed as if every maximum context were allocated at once.
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.native.rope_bytes, required.native.scratch_payload_bound,
                    config.native.allocator_reserve_bytes])?)?;
            let workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, required.workspace_bytes)?;
            let budget = Int4ExtractionMapCohortBudget {
                context_limit: config.native.context_tokens,
                native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound },
                max_workspace_bytes: required.workspace_bytes,
                max_result_bytes: required.max_result_bytes,
            };
            let result = allocate(output, || {
                let result = prepared.execute_cohorts_with_control(&admitted, &model.inner.loaded.value,
                    &input.value.vocabulary, budget, kernel, control).map_err(map_error)?;
                expected.verify_completed(&result.document).map_err(map_error)?;
                poll(control)?;
                Ok(result)
            })?;
            drop(admitted);
            drop(input);
            drop(workspace);
            drop(scratch);
            drop(kv);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn check_peak(required: Int4ExtractionMapCohortRequirement, ceiling: Int4ExtractionMapCohortRequirement)
    -> Result<(), HostedError> {
    if required.native.kv_bytes > ceiling.native.kv_bytes
        || required.native.rope_bytes > ceiling.native.rope_bytes
        || required.native.scratch_payload_bound > ceiling.native.scratch_payload_bound
        || required.workspace_bytes > ceiling.workspace_bytes || required.max_result_bytes > ceiling.max_result_bytes {
        return Err(HostedError::Limits("INT4 extraction complete cohort peak exceeds declaration"));
    }
    Ok(())
}

#[cfg(test)] mod tests;
