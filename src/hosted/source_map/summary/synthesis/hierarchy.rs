//! Hierarchical synthesis on one process-owned model, lease and controller.
use super::*;
use crate::tasks::source_planning::quantized::long::summary::synthesis::hierarchy::{
    Int8SummaryHierarchyRun, SummaryHierarchyLimits,
};

impl NlpEngine {
    /// Explicit lossy quote compression across multiple context-sized levels.
    /// Every native pass and its original-source citations survive in the result.
    /// No new runtime/model, no silent fallback, no generated prose as evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary_hierarchical(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SummaryHierarchyRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        validate_synthesis(&config, source.len(), required.kv_bytes)?;
        hierarchy.validate().map_err(HostedError::SourceMap)?;
        check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
        let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Keep the WHOLE input owner and preparation charge through drain.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("hierarchical summary task context"))?;
            let prepared = input.value.planner.plan_int8_summary_hierarchy_with_control(&input.value.source,
                config.task, hierarchy, config.budget, &context, config.planning, config.mapping, control)
                .map_err(HostedError::SourceMap)?;
            let expected = prepared.preflight_metadata();
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                hierarchy_bytes(config, expected.chunk_count(), hierarchy)?)?;
            let tokens = (expected.chunk_count() as u64).checked_add(hierarchy.max_passes as u64)
                .and_then(|n| n.checked_mul(u64::from(config.budget.max_output_tokens)))
                .ok_or(HostedError::Limits("hierarchical summary token arithmetic"))?;
            // Map AND all possible reduce transcripts, not just the final pass,
            // remain covered by output ownership through serialization/delivery.
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(expected.chunk_count())
                .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                if let Some(cause) = control.prefill_checkpoint(0) {
                    return Err(HostedError::Source(Int8SourceError::Cancelled(cause)));
                }
                check_model_identity(model.artifact_identity(), identity)?;
                constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &input.value.planner,
                    &mut engine.value, &input.value.vocabulary, control).map_err(HostedError::SourceMap)?;
                expected.verify_completed(&result).map_err(HostedError::SourceMap)?;
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
pub(super) fn hierarchy_bytes(config: &SourceMapConfig<SourceSummarySynthesis>, chunks: usize,
    hierarchy: SummaryHierarchyLimits) -> Result<u64, HostedError> {
    hierarchy.validate().map_err(HostedError::SourceMap)?;
    // The existing synthesis reservation includes discovery, one final native
    // payload, evidence copies, prefix tables and the SHARED verification fanout.
    // Extra passes add their retained native payloads. Two frontier generations
    // fit the existing evidence-copy reserve; no per-level ledger is renewed.
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(HostedError::Limits("hierarchical summary memory arithmetic"));
    let per_pass = sum(&[mul(config.budget.max_output_bytes, 4)?,
        mul(u64::from(config.budget.max_output_tokens), 8)?])?;
    sum(&[synthesis_bytes(config, chunks)?, mul(per_pass, (hierarchy.max_passes - 1) as u64)?,
        mul(hierarchy.max_passes as u64, 256)?, mul(hierarchy.max_levels as u64, 128)?])
}
