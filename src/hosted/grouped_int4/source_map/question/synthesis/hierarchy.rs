//! Physically owned question answering across bounded quotation frontiers.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::question::synthesis::hierarchy::{
    Int4QuestionHierarchyRun, QuestionHierarchyLimits, PreparedInt4QuestionHierarchy,
    cohort::Int4QuestionHierarchyCohortRun,
    grouped::Int4QuestionHierarchyGroupedRun,
};
use crate::tasks::source_planning::grouped_int4::long::cohort::hierarchy::Int4HierarchyCohortSchedule;
use native_cohorts::{GroupedHierarchy, GroupedHierarchyPlan};

impl<T: std::borrow::Borrow<SourceQuestionSynthesis>> Int4SourceMapConfig<T> {
    /// Reserve every permitted native answer transcript and complete-level
    /// planning witness. Evidence and verification limits remain shared by the
    /// entire invocation; these checked reserves are not measured RSS claims.
    pub fn question_hierarchy_workspace_bytes(&self, chunks: usize,
        hierarchy: QuestionHierarchyLimits) -> Result<u64, HostedError> {
        self.workspace().question_hierarchy(chunks, self.task.borrow(), hierarchy)
    }
}

impl NlpEngine {
    /// Run independent evidence groups within each fully admitted level using
    /// shared native layers, with an independent choice of discovery schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer_hierarchy_grouped(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        schedule: Int4HierarchyCohortSchedule, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4QuestionHierarchyGroupedRun>, HostedError> {
        native_cohorts::execute(self, model, source, planner, vocabulary, config,
            GroupedHierarchy { limits: hierarchy, schedule }, kernel, cancellation)
    }

    /// Batch independent discovery answers while keeping every dependent
    /// quotation-selection level serial. Original question/evidence semantics,
    /// retained pass transcripts and complete shared limits stay unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer_hierarchy_cohorts(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4QuestionHierarchyCohortRun>, HostedError> {
        native_cohorts::execute(self, model, source, planner, vocabulary, config, hierarchy, kernel, cancellation)
    }

    /// Explicit lossy answer reduction. Every pass receives the original
    /// question and only independently verified original-source quotations.
    /// Intermediate answers and abstentions stay inspectable; generated answer
    /// prose never enters the next evidence frontier. No partial hierarchy is
    /// published on cancellation, refusal, or nonprogress.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer_hierarchy(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4QuestionHierarchyRun>, HostedError> {
        native_synthesis::execute(self, model, source, planner, vocabulary, config,
            hierarchy, kernel, cancellation)
    }

    /// Keep one numerical team through discovery, every evidence-selection
    /// level and the final synthesis pass. Whole-invocation work, evidence,
    /// transcripts and cancellation remain shared with serial execution.
    /// Independent chunk and hierarchy-group cohorts are separate schedules.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer_hierarchy_with_threads(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        kernel: PackedInt4Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4QuestionHierarchyRun>, HostedError> {
        native_synthesis::execute_with_threads(self, model, source, planner, vocabulary, config,
            hierarchy, kernel, compute_threads, cancellation)
    }
}

fn hierarchy_requirements(config: &Int4SourceMapConfig<SourceQuestionSynthesis>,
    hierarchy: QuestionHierarchyLimits, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    let required = synthesis_requirements(config, source_bytes, width, planner)?;
    hierarchy.validate().map_err(map_error)?;
    config.question_hierarchy_workspace_bytes(config.mapping.chunks.max_chunks, hierarchy)?;
    Ok(required)
}

fn hierarchy_output_tokens(per_pass: u32, hierarchy: QuestionHierarchyLimits) -> Result<u64, HostedError> {
    hierarchy.validate().map_err(map_error)?;
    if per_pass == 0 { return Err(HostedError::Limits("INT4 answer hierarchy transcript count")); }
    (hierarchy.max_passes as u64).checked_mul(u64::from(per_pass))
        .ok_or(HostedError::Limits("INT4 answer hierarchy output token arithmetic"))
}

impl native_cohorts::SynthesisMode for QuestionHierarchyLimits {
    type Task = SourceQuestionSynthesis;
    type Output = Int4QuestionHierarchyCohortRun;
    type Prepared<'s> = PreparedInt4QuestionHierarchy<'s>;
    fn requirements(self, config: &Int4SourceMapConfig<Self::Task>, source_bytes: usize,
        width: usize, planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
        hierarchy_requirements(config, self, source_bytes, width, planner)
    }
    fn input_bytes(self, source_capacity: usize, config: &Int4SourceMapConfig<Self::Task>)
        -> Result<u64, HostedError> {
        synthesis_input_bytes(source_capacity, config)
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        config.question_hierarchy_workspace_bytes(chunks, self)
    }
    fn output_tokens(self, config: &Int4SourceMapConfig<Self::Task>, _: usize) -> Result<u64, HostedError> {
        hierarchy_output_tokens(config.budget.max_output_tokens, self)
    }
    fn prepare<'s, C: DecodeStepControl>(self, source: &'s str, planner: &SourceTaskPlanner,
        config: &Int4SourceMapConfig<Self::Task>, context: &PlanContext<'_>, width: usize, control: &mut C)
        -> Result<Self::Prepared<'s>, HostedError> {
        planner.plan_int4_question_hierarchy_with_control(source, &config.task, self, config.budget,
            context, config.planning, config.mapping, width, control).map_err(map_error)
    }
}
impl native_cohorts::SynthesisPlan for PreparedInt4QuestionHierarchy<'_> {
    type Output = Int4QuestionHierarchyCohortRun;
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
        expected.verify_completed(&result.document).map_err(map_error)?;
        Ok(result)
    }
}

impl native_cohorts::SynthesisMode for GroupedHierarchy<QuestionHierarchyLimits> {
    type Task = SourceQuestionSynthesis;
    type Output = Int4QuestionHierarchyGroupedRun;
    type Prepared<'s> = GroupedHierarchyPlan<PreparedInt4QuestionHierarchy<'s>>;
    fn requirements(self, config: &Int4SourceMapConfig<Self::Task>, source_bytes: usize,
        width: usize, planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
        self.schedule.validate(self.limits.max_passes).map_err(map_error)?;
        hierarchy_requirements(config, self.limits, source_bytes, width, planner)
    }
    fn cohort_requirements(self, config: &Int4SourceMapConfig<Self::Task>, width: usize)
        -> Result<Int4SourceMapCohortRequirement, HostedError> {
        config.hierarchy_cohort_requirements(width, self.schedule)
    }
    fn input_bytes(self, source_capacity: usize, config: &Int4SourceMapConfig<Self::Task>)
        -> Result<u64, HostedError> {
        sum(&[synthesis_input_bytes(source_capacity, config)?,
            self.schedule.additional_preparation_bytes(config.preparation_reserve_bytes).map_err(map_error)?])
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        config.question_hierarchy_workspace_bytes(chunks, self.limits)
    }
    fn output_tokens(self, config: &Int4SourceMapConfig<Self::Task>, _: usize) -> Result<u64, HostedError> {
        hierarchy_output_tokens(config.budget.max_output_tokens, self.limits)
    }
    fn prepare<'s, C: DecodeStepControl>(self, source: &'s str, planner: &SourceTaskPlanner,
        config: &Int4SourceMapConfig<Self::Task>, context: &PlanContext<'_>, width: usize, control: &mut C)
        -> Result<Self::Prepared<'s>, HostedError> {
        let prepared = planner.plan_int4_question_hierarchy_with_control(source, &config.task, self.limits,
            config.budget, context, config.planning, config.mapping, width, control).map_err(map_error)?;
        Ok(GroupedHierarchyPlan { prepared, schedule: self.schedule })
    }
}
impl native_cohorts::SynthesisPlan for GroupedHierarchyPlan<PreparedInt4QuestionHierarchy<'_>> {
    type Output = Int4QuestionHierarchyGroupedRun;
    fn group_width(&self) -> usize { self.prepared.group_width() }
    fn source_chunks(&self) -> usize { self.prepared.preflight_metadata().discovery().chunk_count() }
    fn native_chunks(&self) -> usize { self.prepared.preflight_metadata().discovery().native_chunks() }
    fn max_kv_bytes(&self) -> u64 { self.prepared.max_kv_bytes() }
    fn max_result_bytes(&self) -> u64 { self.prepared.max_result_bytes() }
    fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.prepared.execution_identities()
    }
    fn cohort_requirements(&self, context_limit: usize) -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        self.prepared.hierarchy_cohort_requirements(context_limit, self.schedule)
    }
    fn run<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity], planner: &SourceTaskPlanner,
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C) -> Result<Self::Output, HostedError> {
        let expected = self.prepared.preflight_metadata();
        let result = self.prepared.execute_hierarchy_cohorts_with_control(admitted, planner, weights,
            vocabulary, self.schedule, budget, kernel, control).map_err(map_error)?;
        expected.verify_completed(&result.document).map_err(map_error)?;
        Ok(result)
    }
}


impl native_synthesis::SynthesisPlan for PreparedInt4QuestionHierarchy<'_> {
    type Output = Int4QuestionHierarchyRun;
    fn validate_metadata(&self) -> Result<(), HostedError> {
        let expected = self.preflight_metadata();
        let discovery = expected.discovery();
        if discovery.group_width() != self.group_width() || expected.group_width() != self.group_width() {
            return Err(HostedError::ModelIdentity);
        }
        Ok(())
    }
    fn run_on_engine(self, admitted: &[ExecutionIdentity], planner: &SourceTaskPlanner,
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        control: &mut RunControl) -> Result<Self::Output, HostedError> {
        let expected = self.preflight_metadata();
        let result = self.execute_with_control(admitted, planner, engine, vocabulary, control).map_err(map_error)?;
        expected.verify_completed(&result).map_err(map_error)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
