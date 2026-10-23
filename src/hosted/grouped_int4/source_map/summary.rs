//! Complete cited-summary reduction over the same charged INT4 map host.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::summary::{
    Int4CorpusSummaryError, Int4CorpusSummaryRun, Int4SummaryLimits,
};

impl NlpEngine {
    /// Summarize every original chunk and return one deterministic ranked
    /// bullet/evidence union. Every candidate survives intermediate reduction;
    /// only the complete final root may apply the declared bullet cap.
    /// Citations prove source membership, not semantic entailment.
    #[allow(clippy::too_many_arguments)]
    pub fn summarize_int4_source(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, summary: Int4SummaryLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4CorpusSummaryRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let required = map_requirements(&config, source.len(), width, &planner)?;
        if !matches!(&config.task, SourceMapTask::Summarize(_)) {
            return Err(HostedError::Limits("INT4 complete summary requires summary maps"));
        }
        summary.validate(config.mapping).map_err(summary_error)?;
        let bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 summary map task context"))?;
            let prepared = input.value.planner.plan_int4_map_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, width, control)
                .map_err(map_error)?;
            prepared.check_summary(summary).map_err(summary_error)?;
            let expected = prepared.preflight_metadata();
            if prepared.group_width() != width || expected.group_width() != width
                || prepared.execution_identities().len() != prepared.chunk_count()
                || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                config.workspace_bytes(prepared.chunk_count())?)?;
            // The final union omits native token transcripts. The full native
            // frontier remains charged separately until all reduction finishes.
            let output = output_claim(&lease, summary.max_result_bytes as u64, 0)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(prepared.chunk_count())
                .map_err(|_| map_error(Int4SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, model.model_identity())
                    .map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_summary_with_control(&admitted,
                    &mut engine.value, &input.value.vocabulary, summary, control).map_err(summary_error)?;
                expected.verify_summary_completed(&result).map_err(summary_error)?;
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
fn summary_error(error: Int4CorpusSummaryError) -> HostedError {
    HostedError::Int4(Int4HostError::SourceSummary(error))
}
