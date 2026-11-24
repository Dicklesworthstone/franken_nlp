//! One owned INT8 invocation across discovery and all quotation-selection levels.
use super::*;
use crate::tasks::source_planning::quantized::long::question::synthesis::hierarchy::{
    Int8QuestionHierarchyRun, QuestionHierarchyLimits, memory::workspace_bytes};

impl NlpEngine {
    /// Explicit lossy reduction when collected evidence exceeds one answer
    /// context. Every pass retains the original question and verified source
    /// quotes, with all native/lifted results owned through physical completion.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_document_answer_hierarchical(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8QuestionHierarchyRun>, HostedError> {
        self.synthesize_int8_document_answer_hierarchical_scheduled(model, source, planner, vocabulary,
            config, hierarchy, None, None, cancellation)?.serial()
    }

    /// Retain one bounded prompt workspace across every quotation-selection level.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_document_answer_hierarchical_layer_major(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8QuestionHierarchyRun>, HostedError> {
        self.synthesize_int8_document_answer_hierarchical_scheduled(model, source, planner, vocabulary,
            config, hierarchy, Some(prefill), None, cancellation)?.serial()
    }

    /// Reuse one persistent team through discovery and all admitted hierarchy nodes.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_document_answer_hierarchical_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        kernel: Int8Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8QuestionHierarchyRun>, HostedError> {
        self.synthesize_int8_document_answer_hierarchical_scheduled(model, source, planner, vocabulary,
            config, hierarchy, None, Some((kernel, compute_threads)), cancellation)?.team()
    }

    /// Compose workers and prompt morsels under the original complete-document
    /// allowance; dynamic evidence and tokenizer work do not reset between levels.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_document_answer_hierarchical_layer_major_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        kernel: Int8Kernel, compute_threads: usize, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8QuestionHierarchyRun>, HostedError> {
        self.synthesize_int8_document_answer_hierarchical_scheduled(model, source, planner, vocabulary,
            config, hierarchy, Some(prefill), Some((kernel, compute_threads)), cancellation)?.team()
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_int8_document_answer_hierarchical_scheduled(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>, cancellation: CancellationToken)
        -> Result<scheduled::Completion<Int8QuestionHierarchyRun>, HostedError> {
        let required = requirements(config.native)?;
        validate_synthesis(&config, source.len(), required.kv_bytes, planner.document_encoding())?;
        hierarchy.validate().map_err(HostedError::SourceMap)?;
        hierarchy_bytes(&config, config.mapping.chunks.max_chunks, hierarchy)?;
        let bytes = input_bytes(&config, source.capacity(), &planner)?;
        scheduled::run(self, model, SourceMapInput { source, planner, vocabulary, config },
            bytes, prefill, team, cancellation, move |input, invocation, control| {
                let config = &input.config;
                let context = PlanContext::new(&config.identity, config.budget)
                    .map_err(|_| HostedError::Limits("question hierarchy task context"))?;
                let prepared = input.planner.plan_int8_question_hierarchy_with_control(&input.source,
                    &config.task, hierarchy, config.budget, &context, config.planning, config.mapping, control)
                    .map_err(HostedError::SourceMap)?;
                let expected = prepared.preflight_metadata();
                let discovery = expected.discovery();
                let temporary = hierarchy_bytes(config, discovery.chunk_count(), hierarchy)?;
                let tokens = transcript_tokens(config.budget.max_output_tokens, discovery.native_chunks(), hierarchy)?;
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

fn hierarchy_bytes(config: &SourceMapConfig<SourceQuestionSynthesis>, chunks: usize,
    hierarchy: QuestionHierarchyLimits) -> Result<u64, HostedError> {
    workspace_bytes(config.budget, config.mapping, chunks, config.reduction_reserve_bytes,
        config.task.limits, config.task.question.verification, hierarchy).map_err(HostedError::SourceMap)
}
fn transcript_tokens(per_pass: u32, discovery_passes: usize, hierarchy: QuestionHierarchyLimits)
    -> Result<u64, HostedError> {
    hierarchy.validate().map_err(HostedError::SourceMap)?;
    if per_pass == 0 { return Err(HostedError::Limits("question hierarchy output token allowance")); }
    (discovery_passes as u64).checked_add(hierarchy.max_passes as u64)
        .and_then(|n| n.checked_mul(u64::from(per_pass)))
        .ok_or(HostedError::Limits("question hierarchy output token arithmetic"))
}

#[cfg(test)] mod tests;
