//! Bounded evidence-only levels on one physically owned native INT4 engine.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::summary::synthesis::hierarchy::{
    Int4SummaryHierarchyRun, SummaryHierarchyLimits, PreparedInt4SummaryHierarchy,
    cohort::Int4SummaryHierarchyCohortRun,
    grouped::Int4SummaryHierarchyGroupedRun,
};
use crate::tasks::source_planning::grouped_int4::long::cohort::hierarchy::Int4HierarchyCohortSchedule;
use native_cohorts::{GroupedHierarchy, GroupedHierarchyPlan};

impl<T: std::borrow::Borrow<SourceSummarySynthesis>> Int4SourceMapConfig<T> {
    /// Price every permitted native transcript plus complete-level planning
    /// witnesses. Evidence and verification allowances stay shared across all
    /// levels. These are checked admission reserves, not measured RSS bounds.
    pub fn summary_hierarchy_workspace_bytes(&self, chunks: usize,
        hierarchy: SummaryHierarchyLimits) -> Result<u64, HostedError> {
        self.workspace().summary_hierarchy(chunks, self.task.borrow(), hierarchy)
    }
}

impl NlpEngine {
    /// Execute independent groups within each fully admitted hierarchy level
    /// through shared native layers. Discovery batching is separately selected.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_source_summary_hierarchy_grouped(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        schedule: Int4HierarchyCohortSchedule, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SummaryHierarchyGroupedRun>, HostedError> {
        native_cohorts::execute(self, model, source, planner, vocabulary, config,
            GroupedHierarchy { limits: hierarchy, schedule }, kernel, cancellation)
    }

    /// Batch only the independent discovery maps of an explicitly bounded
    /// quotation hierarchy. Every dependent level keeps its serial numerics,
    /// complete evidence frontier, shared work budget and original citations.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_source_summary_hierarchy_cohorts(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SummaryHierarchyCohortRun>, HostedError> {
        native_cohorts::execute(self, model, source, planner, vocabulary, config, hierarchy, kernel, cancellation)
    }

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
        native_synthesis::execute(self, model, source, planner, vocabulary, config,
            hierarchy, kernel, cancellation)
    }

    /// Keep one numerical team through discovery, every evidence-selection
    /// level and the final synthesis pass. Whole-invocation work, evidence,
    /// transcripts and cancellation remain shared with serial execution.
    /// Independent chunk and hierarchy-group cohorts are separate schedules.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_source_summary_hierarchy_with_threads(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceSummarySynthesis>, hierarchy: SummaryHierarchyLimits,
        kernel: PackedInt4Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4SummaryHierarchyRun>, HostedError> {
        native_synthesis::execute_with_threads(self, model, source, planner, vocabulary, config,
            hierarchy, kernel, compute_threads, cancellation)
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

impl native_cohorts::SynthesisMode for SummaryHierarchyLimits {
    type Task = SourceSummarySynthesis;
    type Output = Int4SummaryHierarchyCohortRun;
    type Prepared<'s> = PreparedInt4SummaryHierarchy<'s>;
    fn requirements(self, config: &Int4SourceMapConfig<Self::Task>, source_bytes: usize,
        width: usize, planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
        hierarchy_requirements(config, self, source_bytes, width, planner)
    }
    fn input_bytes(self, source_capacity: usize, config: &Int4SourceMapConfig<Self::Task>)
        -> Result<u64, HostedError> {
        sum(&[source_capacity as u64, config.preparation_reserve_bytes])
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        config.summary_hierarchy_workspace_bytes(chunks, self)
    }
    fn output_tokens(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        hierarchy_output_tokens(chunks, config.budget.max_output_tokens, self)
    }
    fn prepare<'s, C: DecodeStepControl>(self, source: &'s str, planner: &SourceTaskPlanner,
        config: &Int4SourceMapConfig<Self::Task>, context: &PlanContext<'_>, width: usize, control: &mut C)
        -> Result<Self::Prepared<'s>, HostedError> {
        planner.plan_int4_summary_hierarchy_with_control(source, config.task, self, config.budget,
            context, config.planning, config.mapping, width, control).map_err(map_error)
    }
}
impl native_cohorts::SynthesisPlan for PreparedInt4SummaryHierarchy<'_> {
    type Output = Int4SummaryHierarchyCohortRun;
    fn group_width(&self) -> usize { self.group_width() }
    fn source_chunks(&self) -> usize { self.preflight_metadata().chunk_count() }
    fn native_chunks(&self) -> usize { self.preflight_metadata().chunk_count() }
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

impl native_cohorts::SynthesisMode for GroupedHierarchy<SummaryHierarchyLimits> {
    type Task = SourceSummarySynthesis;
    type Output = Int4SummaryHierarchyGroupedRun;
    type Prepared<'s> = GroupedHierarchyPlan<PreparedInt4SummaryHierarchy<'s>>;
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
        sum(&[source_capacity as u64, config.preparation_reserve_bytes,
            self.schedule.additional_preparation_bytes(config.preparation_reserve_bytes).map_err(map_error)?])
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        config.summary_hierarchy_workspace_bytes(chunks, self.limits)
    }
    fn output_tokens(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        hierarchy_output_tokens(chunks, config.budget.max_output_tokens, self.limits)
    }
    fn prepare<'s, C: DecodeStepControl>(self, source: &'s str, planner: &SourceTaskPlanner,
        config: &Int4SourceMapConfig<Self::Task>, context: &PlanContext<'_>, width: usize, control: &mut C)
        -> Result<Self::Prepared<'s>, HostedError> {
        let prepared = planner.plan_int4_summary_hierarchy_with_control(source, config.task, self.limits,
            config.budget, context, config.planning, config.mapping, width, control).map_err(map_error)?;
        Ok(GroupedHierarchyPlan { prepared, schedule: self.schedule })
    }
}
impl native_cohorts::SynthesisPlan for GroupedHierarchyPlan<PreparedInt4SummaryHierarchy<'_>> {
    type Output = Int4SummaryHierarchyGroupedRun;
    fn group_width(&self) -> usize { self.prepared.group_width() }
    fn source_chunks(&self) -> usize { self.prepared.preflight_metadata().chunk_count() }
    fn native_chunks(&self) -> usize { self.prepared.preflight_metadata().chunk_count() }
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
        expected.verify_completed_with_control(&result.document, control).map_err(map_error)?;
        Ok(result)
    }
}


impl native_synthesis::SynthesisPlan for PreparedInt4SummaryHierarchy<'_> {
    type Output = Int4SummaryHierarchyRun;
    fn validate_metadata(&self) -> Result<(), HostedError> {
        if self.preflight_metadata().group_width() != self.group_width() {
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
