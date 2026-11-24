//! Hierarchical synthesis on one process-owned model, lease and controller.
use super::*;
use crate::tasks::source_planning::quantized::long::summary::synthesis::hierarchy::{
    Int8SummaryHierarchyRun, SummaryHierarchyLimits,
};

impl NlpEngine {
    /// Explicit lossy quote compression across multiple context-sized levels.
    /// Every native pass and its original-source citations survive in the result.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary_hierarchical(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8SummaryHierarchyRun>, HostedError> {
        self.synthesize_int8_summary_hierarchical_scheduled(model, source, planner, vocabulary,
            config, hierarchy, None, None, cancellation)?.serial()
    }

    /// One prompt workspace spans discovery and all quotation-reduction levels.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary_hierarchical_layer_major(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SummaryHierarchyRun>, HostedError> {
        self.synthesize_int8_summary_hierarchical_scheduled(model, source, planner, vocabulary,
            config, hierarchy, Some(prefill), None, cancellation)?.serial()
    }

    /// Keep one fixed numerical team across every map and hierarchy pass.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary_hierarchical_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        kernel: Int8Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SummaryHierarchyRun>, HostedError> {
        self.synthesize_int8_summary_hierarchical_scheduled(model, source, planner, vocabulary,
            config, hierarchy, None, Some((kernel, compute_threads)), cancellation)?.team()
    }

    /// Compose prompt morsels with the persistent team while retaining original
    /// source binding, evidence lineage and all nonrenewable operation budgets.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary_hierarchical_layer_major_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        kernel: Int8Kernel, compute_threads: usize, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SummaryHierarchyRun>, HostedError> {
        self.synthesize_int8_summary_hierarchical_scheduled(model, source, planner, vocabulary,
            config, hierarchy, Some(prefill), Some((kernel, compute_threads)), cancellation)?.team()
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_int8_summary_hierarchical_scheduled(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>, cancellation: CancellationToken)
        -> Result<scheduled::Completion<Int8SummaryHierarchyRun>, HostedError> {
        let required = requirements(config.native)?;
        validate_synthesis(&config, source.len(), required.kv_bytes)?;
        hierarchy.validate().map_err(HostedError::SourceMap)?;
        let bytes = input_bytes(&config, source.capacity(), &planner)?;
        scheduled::run(self, model, SourceMapInput { source, planner, vocabulary, config },
            bytes, prefill, team, cancellation, move |input, invocation, control| {
                let config = &input.config;
                let context = PlanContext::new(&config.identity, config.budget)
                    .map_err(|_| HostedError::Limits("hierarchical summary task context"))?;
                let prepared = input.planner.plan_int8_summary_hierarchy_with_control(&input.source,
                    config.task, hierarchy, config.budget, &context, config.planning, config.mapping, control)
                    .map_err(HostedError::SourceMap)?;
                let expected = prepared.preflight_metadata();
                let temporary = hierarchy_bytes(config, expected.chunk_count(), hierarchy)?;
                let tokens = (expected.chunk_count() as u64).checked_add(hierarchy.max_passes as u64)
                    .and_then(|n| n.checked_mul(u64::from(config.budget.max_output_tokens)))
                    .ok_or(HostedError::Limits("hierarchical summary token arithmetic"))?;
                let output_bytes = prepared.max_result_bytes();
                let admitted = scheduled::admit(prepared.execution_identities(),
                    invocation.artifact(), &input.planner, control)?;
                let prefill = invocation.prefill();
                invocation.execute(temporary, output_bytes, tokens, control, |engine, control| {
                    let result = match prefill {
                        Some(prefill) => prepared.execute_layer_major_with_control(&admitted, &input.planner,
                            engine, &input.vocabulary, prefill, control),
                        None => prepared.execute_with_control(&admitted, &input.planner,
                            engine, &input.vocabulary, control),
                    }.map_err(HostedError::SourceMap)?;
                    expected.verify_completed_with_control(&result, control).map_err(HostedError::SourceMap)?;
                    Ok(result)
                })
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
