//! Physically owned discovery and a final evidence-only INT4 answer.
use super::*;
use super::super::cohort::synthesis as native_cohorts;
use super::super::synthesis as native_synthesis;
use crate::tasks::source_planning::grouped_int4::long::{
    cohort::{Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement},
    question::synthesis::{Int4QuestionSynthesisRun, SourceQuestionSynthesis, PreparedInt4QuestionSynthesis,
        cohort::Int4QuestionSynthesisCohortRun},
};
mod hierarchy;

impl<T: std::borrow::Borrow<SourceQuestionSynthesis>> Int4SourceMapConfig<T> {
    /// Charge all source/question maps, verbatim evidence and its origin fanout,
    /// independent occurrence scans and the temporary final native transcript.
    /// The final result keeps text/citations, so native token storage is temporary.
    pub fn question_synthesis_workspace_bytes(&self, chunks: usize) -> Result<u64, HostedError> {
        self.workspace().question_synthesis(chunks, self.task.borrow())
    }
}

impl NlpEngine {
    /// Share native layers across the independent nonblank discovery chunks,
    /// then answer from every verified quotation with the original serial pass.
    /// Private question/source owners and full phase admission survive both.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4QuestionSynthesisCohortRun>, HostedError> {
        native_cohorts::execute(self, model, source, planner, vocabulary, config,
            QuestionSynthesisMode, kernel, cancellation)
    }

    /// Answer a single private question using every collected verbatim citation
    /// from a complete source-document map. Map answers remain inspectable;
    /// they are never presented to the final model as factual source material.
    /// All work, masks and independent verification share nonrenewable limits.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4QuestionSynthesisRun>, HostedError> {
        native_synthesis::execute(self, model, source, planner, vocabulary, config,
            QuestionSynthesisMode, kernel, cancellation)
    }

    /// Keep one numerical team through every discovery chunk and the final
    /// evidence-only synthesis pass. The complete source, verification budgets
    /// and output owner survive physical worker drain. An early abstention or
    /// empty evidence frontier never renews an allowance.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer_with_threads(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4QuestionSynthesisRun>, HostedError> {
        native_synthesis::execute_with_threads(self, model, source, planner, vocabulary, config,
            QuestionSynthesisMode, kernel, compute_threads, cancellation)
    }
}

fn synthesis_requirements(config: &Int4SourceMapConfig<SourceQuestionSynthesis>,
    source_bytes: usize, width: usize, planner: &SourceTaskPlanner)
    -> Result<Int4MemoryRequirement, HostedError> {
    let required = common_requirements(config, source_bytes, width, planner)?;
    if config.identity.task_spec != ANSWER_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.question.validate().map_err(map_error)?;
    let limits = config.task.limits;
    if config.task.question.question.len() > config.planning.max_input_bytes
        || config.task.question.question.len() > config.budget.max_input_tokens as usize
        || !(1..=1024).contains(&limits.max_evidence_passages)
        || limits.max_evidence_passages > config.planning.max_passages
        || limits.max_evidence_bytes == 0 || limits.max_evidence_bytes > config.planning.max_input_bytes {
        return Err(HostedError::Limits("INT4 answer synthesis question or evidence limits"));
    }
    config.question_synthesis_workspace_bytes(config.mapping.chunks.max_chunks)?;
    Ok(required)
}

fn synthesis_input_bytes(source_capacity: usize, config: &Int4SourceMapConfig<SourceQuestionSynthesis>)
    -> Result<u64, HostedError> {
    sum(&[source_capacity as u64, config.task.question.question.capacity() as u64,
        config.preparation_reserve_bytes])
}

#[derive(Clone, Copy)]
struct QuestionSynthesisMode;
impl native_cohorts::SynthesisMode for QuestionSynthesisMode {
    type Task = SourceQuestionSynthesis;
    type Output = Int4QuestionSynthesisCohortRun;
    type Prepared<'s> = PreparedInt4QuestionSynthesis<'s>;
    fn requirements(self, config: &Int4SourceMapConfig<Self::Task>, source_bytes: usize,
        width: usize, planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
        synthesis_requirements(config, source_bytes, width, planner)
    }
    fn input_bytes(self, source_capacity: usize, config: &Int4SourceMapConfig<Self::Task>)
        -> Result<u64, HostedError> {
        synthesis_input_bytes(source_capacity, config)
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        config.question_synthesis_workspace_bytes(chunks)
    }
    fn output_tokens(self, _: &Int4SourceMapConfig<Self::Task>, _: usize) -> Result<u64, HostedError> {
        // Neither discovery nor the flat final answer retains native token IDs.
        Ok(0)
    }
    fn prepare<'s, C: DecodeStepControl>(self, source: &'s str, planner: &SourceTaskPlanner,
        config: &Int4SourceMapConfig<Self::Task>, context: &PlanContext<'_>, width: usize, control: &mut C)
        -> Result<Self::Prepared<'s>, HostedError> {
        planner.plan_int4_question_synthesis_with_control(source, &config.task, config.budget,
            context, config.planning, config.mapping, width, control).map_err(map_error)
    }
}
impl native_cohorts::SynthesisPlan for PreparedInt4QuestionSynthesis<'_> {
    type Output = Int4QuestionSynthesisCohortRun;
    fn group_width(&self) -> usize { self.group_width() }
    fn source_chunks(&self) -> usize { self.preflight_metadata().discovery().chunk_count() }
    fn native_chunks(&self) -> usize { self.preflight_metadata().discovery().native_chunks() }
    fn max_kv_bytes(&self) -> u64 { self.max_kv_bytes() }
    fn max_result_bytes(&self) -> u64 { self.max_result_bytes() }
    fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> { self.execution_identities() }
    fn cohort_requirements(&self, context_limit: usize) -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        self.cohort_requirements(context_limit)
    }
    fn run<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity], planner: &SourceTaskPlanner,
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C) -> Result<Self::Output, HostedError> {
        let expected = self.preflight_metadata();
        let result = self.execute_cohorts_with_control(admitted, planner, weights, vocabulary, budget, kernel, control)
            .map_err(map_error)?;
        expected.verify_completed_with_control(&result.document, control).map_err(map_error)?;
        Ok(result)
    }
}


impl native_synthesis::SynthesisPlan for PreparedInt4QuestionSynthesis<'_> {
    type Output = Int4QuestionSynthesisRun;
    fn validate_metadata(&self) -> Result<(), HostedError> {
        let expected = self.preflight_metadata();
        let discovery = expected.discovery();
        if discovery.group_width() != self.group_width() {
            return Err(HostedError::ModelIdentity);
        }
        Ok(())
    }
    fn run_on_engine(self, admitted: &[ExecutionIdentity], planner: &SourceTaskPlanner,
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        control: &mut RunControl) -> Result<Self::Output, HostedError> {
        let expected = self.preflight_metadata();
        let result = self.execute_with_control(admitted, planner, engine, vocabulary, control).map_err(map_error)?;
        expected.verify_completed_with_control(&result, control).map_err(map_error)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
