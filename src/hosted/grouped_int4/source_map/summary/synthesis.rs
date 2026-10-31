//! One physically owned INT4 invocation for complete maps and quote-only synthesis.
use super::*;
use super::super::cohort::synthesis as native_cohorts;
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
        let mul = |a: u64, b: u64| a.checked_mul(b)
            .ok_or(HostedError::Limits("INT4 summary synthesis workspace arithmetic"));
        let limits = self.task.borrow().limits;
        sum(&[
            self.workspace_bytes(chunks)?,
            mul(self.budget.max_output_bytes, 4)?,
            mul(u64::from(self.budget.max_output_tokens), 8)?,
            mul(limits.max_evidence_bytes as u64, 64)?,
            mul(limits.max_evidence_segments as u64, 128)?,
            mul(limits.verification.max_matches as u64, 128)?,
            mul(limits.verification.max_fields as u64, 128)?,
        ])
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
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let required = synthesis_requirements(&config, source.len(), width, &planner)?;
        let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // The whole charged owner survives every map, dynamic final plan,
            // native drain and evidence lift, including wrapper cancellation.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 summary synthesis task context"))?;
            let prepared = input.value.planner.plan_int4_summary_synthesis_with_control(
                &input.value.source, config.task, config.budget, &context,
                config.planning, config.mapping, width, control).map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let count = expected.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || count == 0 || prepared.execution_identities().len() != count
                || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                config.summary_synthesis_workspace_bytes(count)?)?;
            // All map token transcripts and the possible final transcript are
            // retained in the output after the physical engine has drained.
            let tokens = synthesis_output_tokens(count, config.budget.max_output_tokens)?;
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
        expected.verify_completed(&result.document).map_err(map_error)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
