//! Shared physical ownership for discovery and evidence-synthesis schedules.
//! Closed, statically dispatched modes retain their original planners,
//! verification, task limits and semantic receipts.
use super::*;
use crate::{
    native_engine::{decode::DecodeStepControl, strict_int4::Int4ModelWeights},
    tasks::source_planning::grouped_int4::long::cohort::COHORT_ENVELOPE_BYTES,
};

pub(in crate::hosted::grouped_int4::source_map) trait SynthesisMode: Copy + Send + 'static {
    type Task: Send + 'static;
    type Output: Send + 'static;
    type Prepared<'s>: SynthesisPlan<Output = Self::Output>;

    fn requirements(self, config: &Int4SourceMapConfig<Self::Task>, source_bytes: usize,
        width: usize, planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError>;
    fn cohort_requirements(self, config: &Int4SourceMapConfig<Self::Task>, width: usize)
        -> Result<Int4SourceMapCohortRequirement, HostedError> {
        config.synthesis_cohort_requirements(width)
    }
    fn input_bytes(self, source_capacity: usize, config: &Int4SourceMapConfig<Self::Task>)
        -> Result<u64, HostedError>;
    fn workspace_bytes(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize)
        -> Result<u64, HostedError>;
    fn output_tokens(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize)
        -> Result<u64, HostedError>;
    #[allow(clippy::too_many_arguments)]
    fn prepare<'s, C: DecodeStepControl>(self, source: &'s str, planner: &SourceTaskPlanner,
        config: &Int4SourceMapConfig<Self::Task>, context: &PlanContext<'_>, width: usize, control: &mut C)
        -> Result<Self::Prepared<'s>, HostedError>;
}

#[derive(Clone, Copy)]
pub(in crate::hosted::grouped_int4::source_map) struct GroupedHierarchy<H> {
    pub limits: H,
    pub schedule: Int4HierarchyCohortSchedule,
}
pub(in crate::hosted::grouped_int4::source_map) struct GroupedHierarchyPlan<P> {
    pub prepared: P,
    pub schedule: Int4HierarchyCohortSchedule,
}

pub(in crate::hosted::grouped_int4::source_map) trait SynthesisPlan: Sized {
    type Output: Send + 'static;
    fn group_width(&self) -> usize;
    fn source_chunks(&self) -> usize;
    fn native_chunks(&self) -> usize;
    fn max_kv_bytes(&self) -> u64;
    fn max_result_bytes(&self) -> u64;
    fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity>;
    fn cohort_requirements(&self, context_limit: usize) -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError>;
    /// Execute the sealed plan and check its original complete semantic receipt
    /// before returning the selected phases' physical ledgers to the owner.
    #[allow(clippy::too_many_arguments)]
    fn run<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity], planner: &SourceTaskPlanner,
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C) -> Result<Self::Output, HostedError>;
}

#[allow(clippy::too_many_arguments)]
pub(in crate::hosted::grouped_int4::source_map) fn execute<M: SynthesisMode>(host: &NlpEngine,
    model: &ResidentInt4, source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    config: Int4SourceMapConfig<M::Task>, mode: M, kernel: PackedInt4Kernel, cancellation: CancellationToken)
    -> Result<HostedOutput<M::Output>, HostedError> {
    dispatch::preflight(host, config.native.run)?;
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    constrained_int4::check_model(&config.identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
    let width = model.model_identity().group_width();
    let row = mode.requirements(&config, source.len(), width, &planner)?;
    let ceiling = mode.cohort_requirements(&config, width)?;
    let bytes = mode.input_bytes(source.capacity(), &config)?;
    let run = config.native.run;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
        || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
    let model = model.clone();
    dispatch::run(host, run, cancellation, move |control| {
        // This complete owner keeps private source/question bytes, planners and
        // their charge alive until every physical engine and evidence pass drains.
        let input = input;
        let config = &input.value.config;
        checkpoint(control)?;
        let context = PlanContext::new(&config.identity, config.budget)
            .map_err(|_| HostedError::Limits("INT4 discovery cohort synthesis task context"))?;
        let prepared = mode.prepare(&input.value.source, &input.value.planner, config, &context, width, control)?;
        let count = prepared.source_chunks();
        let native = prepared.native_chunks();
        if prepared.group_width() != width || count == 0 || count > config.mapping.chunks.max_chunks
            || native == 0 || native > count || prepared.execution_identities().len() != native
            || row.kv_bytes > prepared.max_kv_bytes() {
            return Err(HostedError::ModelIdentity);
        }
        // Short discovery windows never narrow the full contexts reserved for
        // dependent scalar passes or independent hierarchy groups.
        let required = prepared.cohort_requirements(config.native.context_tokens).map_err(map_error)?;
        check_peak(required, ceiling)?;
        let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, mode.workspace_bytes(config, count)?)?;
        let output = output_claim(&lease, sum(&[prepared.max_result_bytes(), COHORT_ENVELOPE_BYTES])?,
            mode.output_tokens(config, count)?)?;
        let mut admitted = Vec::new();
        admitted.try_reserve_exact(native).map_err(|_| map_error(Int4SourceMapError::Allocation))?;
        for identity in prepared.execution_identities() {
            checkpoint(control)?;
            constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
            if identity.task_spec != config.identity.task_spec
                || identity.tokenizer_digest != config.identity.tokenizer_digest
                || identity.template_digest != config.identity.template_digest {
                return Err(HostedError::ModelIdentity);
            }
            admitted.push(identity.clone());
        }
        checkpoint(control)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.native.rope_bytes, required.native.scratch_payload_bound,
                config.native.allocator_reserve_bytes])?)?;
        let workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, required.workspace_bytes)?;
        let budget = Int4SourceMapCohortBudget { context_limit: config.native.context_tokens,
            native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound },
            max_workspace_bytes: required.workspace_bytes, max_result_bytes: required.max_result_bytes };
        let result = allocate(output, || {
            let result = prepared.run(&admitted, &input.value.planner, &model.inner.loaded.value,
                &input.value.vocabulary, budget, kernel, control)?;
            checkpoint(control)?;
            Ok(result)
        })?;
        drop(admitted);
        drop(input);
        drop(workspace);
        drop(scratch);
        drop(kv);
        drop(reduction);
        drop(lease);
        Ok(GuardedOutput::new(result.value, result._memory))
    })
}

#[cfg(test)] mod tests;
