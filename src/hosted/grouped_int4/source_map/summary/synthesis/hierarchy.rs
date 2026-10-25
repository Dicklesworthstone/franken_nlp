//! Bounded evidence-only levels on one physically owned native INT4 engine.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::summary::synthesis::hierarchy::{
    Int4SummaryHierarchyRun, SummaryHierarchyLimits,
};

impl Int4SourceMapConfig<SourceSummarySynthesis> {
    /// Price every permitted native transcript plus complete-level planning
    /// witnesses. Evidence and verification allowances stay shared across all
    /// levels. These are checked admission reserves, not measured RSS bounds.
    pub fn summary_hierarchy_workspace_bytes(&self, chunks: usize,
        hierarchy: SummaryHierarchyLimits) -> Result<u64, HostedError> {
        hierarchy.validate().map_err(map_error)?;
        let mul = |a: u64, b: u64| a.checked_mul(b)
            .ok_or(HostedError::Limits("INT4 summary hierarchy workspace arithmetic"));
        let per_pass = sum(&[mul(self.budget.max_output_bytes, 4)?,
            mul(u64::from(self.budget.max_output_tokens), 8)?])?;
        // The single-pass quote includes discovery, one final native payload,
        // source copies, verifier fanout and two simultaneous evidence frontiers.
        // Compact witnesses bind every group before any forward in its level.
        sum(&[self.summary_synthesis_workspace_bytes(chunks)?,
            mul(per_pass, (hierarchy.max_passes - 1) as u64)?,
            mul(hierarchy.max_passes as u64, 8192 + 256)?,
            mul(hierarchy.max_levels as u64, 128)?])
    }
}

impl NlpEngine {
    /// Explicit lossy quotation selection over context-sized levels. Every
    /// level consumes its complete ordered frontier; only source-verified
    /// quotations may enter the next one. All native passes and their original
    /// citations survive. Failure or nonprogress never publishes a partial tree.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_source_summary_hierarchy(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SummaryHierarchyRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let required = hierarchy_requirements(&config, hierarchy, source.len(), width, &planner)?;
        let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // The source, planners and their full charge remain owned through
            // every level, cancellation, final evidence lift and native drain.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 summary hierarchy task context"))?;
            let prepared = input.value.planner.plan_int4_summary_hierarchy_with_control(
                &input.value.source, config.task, hierarchy, config.budget, &context,
                config.planning, config.mapping, width, control).map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let count = expected.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || count == 0 || prepared.execution_identities().len() != count
                || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                config.summary_hierarchy_workspace_bytes(count, hierarchy)?)?;
            let tokens = hierarchy_output_tokens(count, config.budget.max_output_tokens, hierarchy)?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count).map_err(|_| map_error(Int4SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, model.model_identity())
                    .map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound,
                    config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.kv_bytes,
                        max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &input.value.planner,
                    &mut engine.value, &input.value.vocabulary, control).map_err(map_error)?;
                expected.verify_completed(&result).map_err(map_error)?;
                check_clean(&engine.value)?;
                checkpoint(control)?;
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

fn hierarchy_requirements(config: &Int4SourceMapConfig<SourceSummarySynthesis>,
    hierarchy: SummaryHierarchyLimits, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    let required = synthesis_requirements(config, source_bytes, width, planner)?;
    hierarchy.validate().map_err(map_error)?;
    config.summary_hierarchy_workspace_bytes(config.mapping.chunks.max_chunks, hierarchy)?;
    Ok(required)
}

fn hierarchy_output_tokens(chunks: usize, per_pass: u32, hierarchy: SummaryHierarchyLimits)
    -> Result<u64, HostedError> {
    hierarchy.validate().map_err(map_error)?;
    if !(1..=256).contains(&chunks) || per_pass == 0 {
        return Err(HostedError::Limits("INT4 summary hierarchy transcript count"));
    }
    (chunks as u64).checked_add(hierarchy.max_passes as u64)
        .and_then(|count| count.checked_mul(u64::from(per_pass)))
        .ok_or(HostedError::Limits("INT4 summary hierarchy output token arithmetic"))
}

#[cfg(test)]
mod tests;
