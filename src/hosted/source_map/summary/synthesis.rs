//! One charged hosted invocation for discovery plus evidence-only synthesis.
use super::*;
mod hierarchy;
use crate::tasks::{summarize::SUMMARIZE_TASK_VERSION,
    source_planning::quantized::long::summary::synthesis::{SourceSummarySynthesis, Int8SummarySynthesisRun}};

impl NlpEngine {
    /// Synthesize a new cited summary from all collected verbatim map evidence.
    /// Generated map text never becomes source; every retained native result
    /// and original-source citation stays bound to the complete request.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SummarySynthesisRun>, HostedError> {
        self.synthesize_int8_summary_scheduled(model, source, planner, vocabulary,
            config, None, None, cancellation)?.serial()
    }

    /// Use bounded prompts through discovery and evidence-only summary synthesis.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary_layer_major(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SummarySynthesisRun>, HostedError> {
        self.synthesize_int8_summary_scheduled(model, source, planner, vocabulary,
            config, Some(prefill), None, cancellation)?.serial()
    }

    /// A single admitted projection team serves all map chunks and the final pass.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt8TeamResult<Int8SummarySynthesisRun>, HostedError> {
        self.synthesize_int8_summary_scheduled(model, source, planner, vocabulary,
            config, None, Some((kernel, compute_threads)), cancellation)?.team()
    }

    /// Compose worker and prompt storage without changing any source/reducer
    /// contract or renewing the original native and tokenizer allowances.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_summary_layer_major_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SummarySynthesisRun>, HostedError> {
        self.synthesize_int8_summary_scheduled(model, source, planner, vocabulary,
            config, Some(prefill), Some((kernel, compute_threads)), cancellation)?.team()
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_int8_summary_scheduled(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceSummarySynthesis>, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, cancellation: CancellationToken)
        -> Result<scheduled::Completion<Int8SummarySynthesisRun>, HostedError> {
        let required = requirements(config.native)?;
        validate_synthesis(&config, source.len(), required.kv_bytes)?;
        let bytes = input_bytes(&config, source.capacity(), &planner)?;
        scheduled::run(self, model, SourceMapInput { source, planner, vocabulary, config },
            bytes, prefill, team, cancellation, |input, invocation, control| {
                let config = &input.config;
                let context = PlanContext::new(&config.identity, config.budget)
                    .map_err(|_| HostedError::Limits("summary synthesis task context"))?;
                let prepared = input.planner.plan_int8_summary_synthesis_with_control(&input.source,
                    config.task, config.budget, &context, config.planning, config.mapping, control)
                    .map_err(HostedError::SourceMap)?;
                let expected = prepared.preflight_metadata();
                let temporary = synthesis_bytes(config, expected.chunk_count())?;
                let tokens = (expected.chunk_count() as u64).checked_add(1)
                    .and_then(|n| n.checked_mul(u64::from(config.budget.max_output_tokens)))
                    .ok_or(HostedError::Limits("summary synthesis token arithmetic"))?;
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
fn validate_synthesis(config: &SourceMapConfig<SourceSummarySynthesis>, source_bytes: usize, kv_bytes: u64)
    -> Result<(), HostedError> {
    validate_common(config, source_bytes, kv_bytes)?;
    if config.identity.task_spec != SUMMARIZE_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.validate(config.planning).map_err(HostedError::SourceMap)
}
fn synthesis_bytes(config: &SourceMapConfig<SourceSummarySynthesis>, chunks: usize) -> Result<u64, HostedError> {
    let base = reduction_bytes(config, chunks)?;
    let l = config.task.limits;
    // Existing map frontier + complete final native payload + bounded evidence
    // copies, KMP prefix storage, origin tables and overlapping occurrence fanout.
    // These are modeled reservation inputs, not allocator-enforced RSS bounds.
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(HostedError::Limits("summary evidence arithmetic"));
    sum(&[base, mul(config.budget.max_output_bytes, 4)?,
        mul(u64::from(config.budget.max_output_tokens), 8)?,
        mul(l.max_evidence_bytes as u64, 64)?, mul(l.max_evidence_segments as u64, 128)?,
        mul(l.verification.max_matches as u64, 128)?, mul(l.verification.max_fields as u64, 128)?])
}
#[cfg(test)] mod tests;
