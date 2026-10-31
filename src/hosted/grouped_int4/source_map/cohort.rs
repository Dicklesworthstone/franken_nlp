//! Complete original-source maps with bounded native chunk cohorts.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::cohort::{
    Int4SourceMapCohortBudget, Int4SourceMapCohortMemoryLimits,
    Int4SourceMapCohortRequirement, Int4SourceMapCohortRun,
};
mod aggregate;

impl Int4SourceMapConfig {
    /// Pre-IO peak for a simultaneous raw source-map window. Full document
    /// preparation, frontier, evidence and output remain separately funded.
    pub fn cohort_requirements(&self, group_width: usize)
        -> Result<Int4SourceMapCohortRequirement, HostedError> {
        let chunks = self.mapping.reduction.map_batch_chunks;
        if chunks > self.mapping.chunks.max_chunks {
            return Err(HostedError::Limits("INT4 source cohort exceeds document chunk ceiling"));
        }
        Int4SourceMapCohortMemoryLimits {
            max_chunks: chunks,
            context_tokens: self.native.context_tokens,
            max_output_tokens_per_chunk: u64::from(self.budget.max_output_tokens),
            max_result_bytes_per_chunk: self.budget.max_output_bytes,
        }.requirements(group_width).map_err(map_error)
    }
}

impl NlpEngine {
    /// Map all NER, keyphrase or cited-summary chunks using native shared
    /// projections. Each chunk retains its source grammar, causal context,
    /// semantic finalizer and independently checked original-source spans.
    ///
    /// Preparation and every row's identity/work admission complete before the
    /// first forward. One deadline and whole-document allowances govern all
    /// windows; failure in a later chunk or reduction yields no document result.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_source_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SourceMapCohortRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let row = map_requirements(&config, source.len(), width, &planner)?;
        let ceiling = config.cohort_requirements(width)?;
        let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            checkpoint(control)?;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 source map cohort task context"))?;
            let prepared = input.value.planner.plan_int4_map_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, width, control)
                .map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let count = prepared.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || prepared.execution_identities().len() != count || row.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let required = prepared.cohort_requirements(config.native.context_tokens).map_err(map_error)?;
            check_peak(required, ceiling)?;
            let tokens = (count as u64).checked_mul(u64::from(config.budget.max_output_tokens))
                .ok_or(HostedError::Limits("INT4 source map cohort output tokens"))?;
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, config.workspace_bytes(count)?)?;
            let output = output_claim(&lease, sum(&[prepared.max_result_bytes(), 4096])?, tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count).map_err(|_| map_error(Int4SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, model.model_identity())
                    .map_err(|_| HostedError::ModelIdentity)?;
                if identity.task_spec != config.identity.task_spec
                    || identity.tokenizer_digest != config.identity.tokenizer_digest
                    || identity.template_digest != config.identity.template_digest {
                    return Err(HostedError::ModelIdentity);
                }
                admitted.push(identity.clone());
            }
            checkpoint(control)?;
            // Retain the complete peak while exact-sized native engines are
            // constructed and physically drained for successive ragged windows.
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.native.rope_bytes, required.native.scratch_payload_bound,
                    config.native.allocator_reserve_bytes])?)?;
            let workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, required.workspace_bytes)?;
            let budget = Int4SourceMapCohortBudget {
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
                checkpoint(control)?;
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

fn check_peak(required: Int4SourceMapCohortRequirement, ceiling: Int4SourceMapCohortRequirement)
    -> Result<(), HostedError> {
    if required.native.kv_bytes > ceiling.native.kv_bytes || required.native.rope_bytes > ceiling.native.rope_bytes
        || required.native.scratch_payload_bound > ceiling.native.scratch_payload_bound
        || required.workspace_bytes > ceiling.workspace_bytes || required.max_result_bytes > ceiling.max_result_bytes {
        return Err(HostedError::Limits("INT4 source complete cohort peak exceeds declaration"));
    }
    Ok(())
}

#[cfg(test)] mod tests;
