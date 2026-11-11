//! One physically owned INT4 invocation for complete maps and quote-only synthesis.
use super::*;
use super::super::cohort::synthesis as native_cohorts;
use super::super::synthesis as native_synthesis;
use crate::tasks::{
    source_planning::grouped_int4::long::{
        cohort::{Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement},
        summary::synthesis::{Int4SummarySynthesisRun, SourceSummarySynthesis, PreparedInt4SummarySynthesis,
            cohort::Int4SummarySynthesisCohortRun},
    },
    summarize::SUMMARIZE_TASK_VERSION,
};
mod hierarchy;

impl<T: std::borrow::Borrow<SourceSummarySynthesis>> Int4SourceMapConfig<T> {
    /// Complete discovery frontier, one final native transcript, quote copies,
    /// prefix tables and original-coordinate occurrence fanout. This checked
    /// admission quote is shared with CLI preflight; it is not measured RSS.
    pub fn summary_synthesis_workspace_bytes(&self, chunks: usize) -> Result<u64, HostedError> {
        self.workspace().summary_synthesis(chunks, self.task.borrow())
    }
}

impl NlpEngine {
    /// Share native layers across independent discovery chunks, then synthesize
    /// from the complete verified quotation set with the original serial pass.
    /// One invocation retains full phase admission and discovery-only statistics.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_source_summary_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceSummarySynthesis>, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4SummarySynthesisCohortRun>, HostedError> {
        native_cohorts::execute(self, model, source, planner, vocabulary, config,
            SummarySynthesisMode, kernel, cancellation)
    }

    /// Produce a new cited summary from the complete set of verified verbatim
    /// map quotations. Generated map prose never becomes source evidence.
    /// The final pass is reserved before mapping on all five model-work axes;
    /// no early completion replenishes that allowance. Every collected quote
    /// must fit the final bounded context or the whole request refuses.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_source_summary(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceSummarySynthesis>, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4SummarySynthesisRun>, HostedError> {
        native_synthesis::execute(self, model, source, planner, vocabulary, config,
            SummarySynthesisMode, kernel, cancellation)
    }

    /// Keep one numerical team through every discovery chunk and the final
    /// evidence-only synthesis pass. The complete source, verification budgets
    /// and output owner survive physical worker drain. An early abstention or
    /// empty evidence frontier never renews an allowance.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_source_summary_with_threads(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceSummarySynthesis>, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4SummarySynthesisRun>, HostedError> {
        native_synthesis::execute_with_threads(self, model, source, planner, vocabulary, config,
            SummarySynthesisMode, kernel, compute_threads, cancellation)
    }
}

fn synthesis_requirements(config: &Int4SourceMapConfig<SourceSummarySynthesis>,
    source_bytes: usize, width: usize, planner: &SourceTaskPlanner)
    -> Result<Int4MemoryRequirement, HostedError> {
    let required = common_requirements(config, source_bytes, width, planner)?;
    if config.identity.task_spec != SUMMARIZE_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.validate(config.planning).map_err(map_error)?;
    config.summary_synthesis_workspace_bytes(config.mapping.chunks.max_chunks)?;
    Ok(required)
}

fn synthesis_output_tokens(chunks: usize, per_pass: u32) -> Result<u64, HostedError> {
    if !(1..=256).contains(&chunks) || per_pass == 0 {
        return Err(HostedError::Limits("INT4 summary synthesis output transcript count"));
    }
    (chunks as u64).checked_add(1).and_then(|count| count.checked_mul(u64::from(per_pass)))
        .ok_or(HostedError::Limits("INT4 summary synthesis output token arithmetic"))
}

#[derive(Clone, Copy)]
struct SummarySynthesisMode;
impl native_cohorts::SynthesisMode for SummarySynthesisMode {
    type Task = SourceSummarySynthesis;
    type Output = Int4SummarySynthesisCohortRun;
    type Prepared<'s> = PreparedInt4SummarySynthesis<'s>;
    fn requirements(self, config: &Int4SourceMapConfig<Self::Task>, source_bytes: usize,
        width: usize, planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
        synthesis_requirements(config, source_bytes, width, planner)
    }
    fn input_bytes(self, source_capacity: usize, config: &Int4SourceMapConfig<Self::Task>)
        -> Result<u64, HostedError> {
        sum(&[source_capacity as u64, config.preparation_reserve_bytes])
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        config.summary_synthesis_workspace_bytes(chunks)
    }
    fn output_tokens(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize) -> Result<u64, HostedError> {
        synthesis_output_tokens(chunks, config.budget.max_output_tokens)
    }
    fn prepare<'s, C: DecodeStepControl>(self, source: &'s str, planner: &SourceTaskPlanner,
        config: &Int4SourceMapConfig<Self::Task>, context: &PlanContext<'_>, width: usize, control: &mut C)
        -> Result<Self::Prepared<'s>, HostedError> {
        planner.plan_int4_summary_synthesis_with_control(source, config.task, config.budget,
            context, config.planning, config.mapping, width, control).map_err(map_error)
    }
}
impl native_cohorts::SynthesisPlan for PreparedInt4SummarySynthesis<'_> {
    type Output = Int4SummarySynthesisCohortRun;
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


impl native_synthesis::SynthesisPlan for PreparedInt4SummarySynthesis<'_> {
    type Output = Int4SummarySynthesisRun;
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
