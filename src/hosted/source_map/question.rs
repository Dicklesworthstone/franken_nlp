//! Question-aware document QA on the existing charged process-owned host.
//! Independent answers remain the default; synthesis has its own explicit API.
use super::*;
use crate::{tasks::{answer::ANSWER_TASK_VERSION,
    source_planning::quantized::long::question::{SourceQuestion, Int8QuestionRun}},
    tokenizer::untrusted::UntrustedEncoding};
#[cfg(test)] use crate::tasks::source_planning::quantized::Int8SourceError;
mod synthesis;

impl NlpEngine {
    /// Apply one question to every nonblank source chunk and retain all passage
    /// answers with original-document citations. Source and question ownership
    /// survives complete execution, cancellation and native resource drain.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int8_document(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestion>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8QuestionRun>, HostedError> {
        self.answer_int8_document_scheduled(model, source, planner, vocabulary, config,
            None, None, cancellation)?.serial()
    }

    /// Reuse one bounded prompt workspace across all complete passage answers.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int8_document_layer_major(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestion>, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8QuestionRun>, HostedError> {
        self.answer_int8_document_scheduled(model, source, planner, vocabulary, config,
            Some(prefill), None, cancellation)?.serial()
    }

    /// Keep one admitted projection team across the complete original document.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int8_document_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestion>, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt8TeamResult<Int8QuestionRun>, HostedError> {
        self.answer_int8_document_scheduled(model, source, planner, vocabulary, config,
            None, Some((kernel, compute_threads)), cancellation)?.team()
    }

    /// Compose prompt morsels and persistent workers without renewing source,
    /// model, tokenizer or verification budgets at passage boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int8_document_layer_major_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestion>, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8QuestionRun>, HostedError> {
        self.answer_int8_document_scheduled(model, source, planner, vocabulary, config,
            Some(prefill), Some((kernel, compute_threads)), cancellation)?.team()
    }

    #[allow(clippy::too_many_arguments)]
    fn answer_int8_document_scheduled(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestion>, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, cancellation: CancellationToken)
        -> Result<scheduled::Completion<Int8QuestionRun>, HostedError> {
        let required = requirements(config.native)?;
        validate_question(&config, source.len(), required.kv_bytes, planner.document_encoding())?;
        let bytes = input_bytes(source.capacity(), &config, &planner)?;
        scheduled::run(self, model, SourceMapInput { source, planner, vocabulary, config },
            bytes, prefill, team, cancellation, |input, invocation, control| {
                let config = &input.config;
                let context = PlanContext::new(&config.identity, config.budget)
                    .map_err(|_| HostedError::Limits("document question task context"))?;
                let prepared = input.planner.plan_int8_question_with_control(&input.source,
                    &config.task, config.budget, &context, config.planning, config.mapping, control)
                    .map_err(HostedError::SourceMap)?;
                let expected = prepared.preflight_metadata();
                // Every chunk remains funded, including whitespace ranges and
                // transient native payloads that are lifted into source answers.
                let temporary = reduction_bytes(config, expected.chunk_count())?;
                let output_bytes = prepared.max_result_bytes();
                let admitted = scheduled::admit(prepared.execution_identities(),
                    invocation.artifact(), &input.planner, control)?;
                let prefill = invocation.prefill();
                invocation.execute(temporary, output_bytes, 0, control, |engine, control| {
                    match prefill {
                        Some(prefill) => prepared.execute_layer_major_with_control(&admitted,
                            engine, &input.vocabulary, prefill, control),
                        None => prepared.execute_with_control(&admitted, engine, &input.vocabulary, control),
                    }.map_err(HostedError::SourceMap)
                })
            })
    }
}
fn validate_question(config: &SourceMapConfig<SourceQuestion>, source_bytes: usize, kv_bytes: u64,
    encoding: &UntrustedEncoding)
    -> Result<(), HostedError> {
    validate_common(config, source_bytes, kv_bytes)?;
    if config.identity.task_spec != ANSWER_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.validate().map_err(HostedError::SourceMap)?;
    let byte_limit = match encoding {
        UntrustedEncoding::ByteFallback => config.planning.max_input_bytes.min(config.budget.max_input_tokens as usize),
        UntrustedEncoding::SafeBpeV1(policy) => config.planning.max_input_bytes.min(policy.max_input_bytes),
    };
    if config.task.question.len() > byte_limit {
        return Err(HostedError::Limits("document question input budget"));
    }
    Ok(())
}
fn input_bytes(source_capacity: usize, config: &SourceMapConfig<SourceQuestion>, planner: &SourceTaskPlanner)
    -> Result<u64, HostedError> {
    let encoding = planner.map_document_encoding_workspace_bytes()
        .map_err(|_| HostedError::Limits("INT8 question map encoding admission"))?;
    sum(&[source_capacity as u64, config.task.question.capacity() as u64,
        config.preparation_reserve_bytes, encoding])
}

#[cfg(test)] mod tests;
