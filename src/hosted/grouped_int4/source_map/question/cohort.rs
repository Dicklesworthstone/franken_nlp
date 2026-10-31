//! Native passage cohorts over one complete question-aware source partition.
//! Whitespace remains in the ordered document without native rows or work.
use super::*;
use super::super::cohort::check_peak;
use crate::tasks::source_planning::grouped_int4::long::{
    cohort::Int4SourceMapCohortBudget,
    question::cohort::Int4QuestionCohortRun,
};

impl NlpEngine {
    /// Answer one immutable question over all nonblank source chunks using
    /// exact-sized native cohorts. Preserve every independent answer,
    /// abstention and whitespace range, with original-document citations.
    /// This does not select, vote on or synthesize a global answer.
    ///
    /// The configured map width counts source chunks; physical rows count
    /// only nonblank passages. All-whitespace windows perform no native work.
    /// All plans and complete-document budgets are admitted before inference.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int4_document_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestion>, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4QuestionCohortRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let row = question_requirements(&config, source.len(), width, &planner)?;
        let ceiling = config.cohort_requirements(width)?;
        let bytes = input_bytes(source.capacity(), &config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Retain the full private question/source owner until all native
            // rows, citation scans and complete-document verification finish.
            let input = input;
            let config = &input.value.config;
            checkpoint(control)?;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 document question cohort context"))?;
            let prepared = input.value.planner.plan_int4_question_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, width, control)
                .map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            if prepared.group_width() != width || expected.group_width() != width
                || expected.native_chunks() == 0
                || prepared.execution_identities().len() != expected.native_chunks()
                || row.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let required = prepared.cohort_requirements(config.native.context_tokens).map_err(map_error)?;
            check_peak(required, ceiling)?;
            // Logical whitespace chunks retain source geometry and result
            // storage even when they do not occupy a native cohort row.
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                config.question_workspace_bytes(expected.chunk_count())?)?;
            // Passage answers omit native token transcripts. Transient row
            // results remain funded by the complete map/frontier allowance.
            let output = output_claim(&lease, sum(&[prepared.max_result_bytes(), 4096])?, 0)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(expected.native_chunks())
                .map_err(|_| map_error(Int4SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, model.model_identity())
                    .map_err(|_| HostedError::ModelIdentity)?;
                if identity.task_spec != config.identity.task_spec
                    || identity.tokenizer_digest != config.identity.tokenizer_digest
                    || identity.template_digest != config.identity.template_digest {
                    return Err(HostedError::ModelIdentity);
                }
                admitted.push(identity.clone());
            }
            checkpoint(control)?;
            // These componentwise peaks stay charged across every physical
            // engine's lifetime, including ragged and sparse native windows.
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
                let result = prepared.execute_cohorts_with_control(&admitted, &model.inner.loaded.value,
                    &input.value.vocabulary, budget, kernel, control).map_err(map_error)?;
                expected.verify_completed(&result.document).map_err(map_error)?;
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
}

#[cfg(test)] mod tests;
