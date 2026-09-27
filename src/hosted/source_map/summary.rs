//! Cited whole-document reduction on the same process-owned source-map host.
//! Exact evidence union/ranking only: no second model, runtime or neural reduce.
use super::*;
use crate::{
    corpus::summarize::CorpusSummaryError,
    native_engine::decode::DecodeStepControl,
    tasks::{mapreduce::ExecutionError, source_planning::quantized::{Int8SourceError,
        long::summary::{Int8CorpusSummaryError, Int8CorpusSummaryRun, Int8SummaryLimits}}},
};

impl NlpEngine {
    /// Map all original chunks with the resident INT8 model and produce one
    /// cited, ranked summary. SourceMapConfig fixes the native SummaryOptions;
    /// summary fixes complete-union/evidence bounds and the final bullet cap.
    /// Every native plan is checked before inference, and final top-k is applied
    /// only after all maps and exact-evidence reductions succeed.
    #[allow(clippy::too_many_arguments)]
    pub fn summarize_int8_source(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig, summary: Int8SummaryLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8CorpusSummaryRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        validate(&config, source.len(), required.kv_bytes)?;
        validate_summary(&config.task, config.mapping, summary)?;
        check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
        let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Capture the WHOLE charged input. Source, planners and temporary
            // compilation storage remain charged until all borrowers drain.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("summary map task context"))?;
            let prepared = input.value.planner.plan_int8_map_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, control)
                .map_err(HostedError::SourceMap)?;
            prepared.check_summary(summary).map_err(execution_error)?;
            // Reserve the complete native/evidence frontier, NOT merely final
            // top-k. Intermediate token buffers and copied citations are priced
            // by the existing conservative source-map payload calculation.
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                reduction_bytes(config, prepared.chunk_count())?)?;
            // No generated-token transcripts survive in this final envelope.
            let output = output_claim(&lease, summary.max_result_bytes as u64, 0)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(prepared.chunk_count())
                .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                if let Some(cause) = control.prefill_checkpoint(0) {
                    return Err(HostedError::Source(Int8SourceError::Cancelled(cause)));
                }
                check_model_identity(model.artifact_identity(), identity)?;
                constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || prepared.execute_summary_with_control(&admitted,
                &mut engine.value, &input.value.vocabulary, summary, control).map_err(execution_error))?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn validate_summary(task: &SourceMapTask, mapping: Int8SourceMapLimits, summary: Int8SummaryLimits)
    -> Result<(), HostedError> {
    if !matches!(task, SourceMapTask::Summarize(_)) {
        return Err(HostedError::Limits("complete summary requires native summary maps"));
    }
    summary.validate(mapping).map_err(execution_error)
}
// Preserve actual source/cancellation/evidence causes through the established
// host categories. No arbitrary nested diagnostic strings become public logs.
fn execution_error(error: Int8CorpusSummaryError) -> HostedError {
    match error {
        Int8CorpusSummaryError::Map(e) => HostedError::SourceMap(e),
        Int8CorpusSummaryError::Source(e) => HostedError::Source(e),
        Int8CorpusSummaryError::Summary(e) => HostedError::Source(e.into()),
        Int8CorpusSummaryError::Execution(ExecutionError::Task { source, .. }) => match source {
            CorpusSummaryError::Pass(e) => HostedError::Source(e),
            CorpusSummaryError::Summary(e) => HostedError::Source(e.into()),
            CorpusSummaryError::Poisoned => HostedError::Source(Int8SourceError::InvalidResult),
        },
        Int8CorpusSummaryError::Execution(ExecutionError::Checkpoint(e)) => HostedError::SourceMap(Int8SourceMapError::Chunk(e)),
        Int8CorpusSummaryError::Execution(ExecutionError::AllocationRefused) => HostedError::SourceMap(Int8SourceMapError::Allocation),
        Int8CorpusSummaryError::Execution(ExecutionError::Serialization | ExecutionError::Invariant
            | ExecutionError::InvalidBatch | ExecutionError::InvalidPolicy)
            | Int8CorpusSummaryError::Accounting => HostedError::Source(Int8SourceError::InvalidResult),
        Int8CorpusSummaryError::Execution(_) | Int8CorpusSummaryError::InvalidLimits => HostedError::Limits("complete summary reduction limits"),
    }
}

#[cfg(test)] mod tests;
