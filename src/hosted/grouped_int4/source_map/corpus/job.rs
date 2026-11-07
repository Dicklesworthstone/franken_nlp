//! Authenticated per-record questions reuse the complete native QA processor.
use super::*;
use crate::{
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobLimits, JobProgress,
        runner::{JobRunError, source::grouped_int4::question_document::{
            Int4QuestionDocumentJobPlanner, Int4QuestionDocumentJobProcessor,
            memory::{Int4QuestionDocumentJobMemoryLimits, Int4QuestionDocumentJobMemoryRequirement}}}},
};

struct QuestionJobInput<R> {
    preparation: Option<(Int4QuestionDocumentJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}

/// Quote the same complete population, question/source frontiers, native
/// phase peak and guarded output used by the owned execution entrypoint.
#[allow(clippy::too_many_arguments)]
pub fn int4_question_document_job_requirements(width: usize,
    config: Int4QuestionDocumentBatchConfig, job: JobLimits, host: JobHostLimits,
    reduction_reserve_bytes: u64, cohort_chunks: Option<usize>, hierarchy_groups: Option<usize>)
    -> Result<Int4QuestionDocumentJobMemoryRequirement, HostedJobError> {
    let population = host.required_buffer_bytes(job)?;
    let required = Int4QuestionDocumentJobMemoryLimits {
        document: Int4QuestionDocumentMemoryLimits { context_tokens: host.native.context_tokens,
            allocator_reserve_bytes: host.native.allocator_reserve_bytes,
            preparation_reserve_bytes: host.preparation_reserve_bytes,
            io_reserve_bytes: host.io_reserve_bytes, reduction_reserve_bytes,
            cohort_chunks, hierarchy_groups },
        transport: host.transport, journal_reserve_bytes: host.journal_reserve_bytes,
        serialization_reserve_bytes: host.serialization_reserve_bytes,
    }.requirements(width, config, job).map_err(HostedError::BatchSetup)?;
    if required.population_buffer_bytes != population {
        return Err(HostedError::Limits("question document job population quotation").into());
    }
    Ok(required)
}

impl NlpEngine {
    /// Start or resume complete original sources with a distinct exact private
    /// question per item. Independent answers, flat synthesis and explicitly
    /// enabled hierarchy all share the live corpus's sealed native compiler.
    ///
    /// The complete population authenticates before journal tail repair or
    /// native execution. Only complete source/question results enter the
    /// spool-first, journal-second commit path; failed attempts retain all
    /// five native work charges and masks across process restarts. The fixed
    /// recipe also proves both verification pools against the lifetime attempt
    /// ceiling. No chunk or hierarchy pass is a durable checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_question_documents<R>(&self, model: &ResidentInt4,
        planner: Int4QuestionDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, host: JobHostLimits, reduction_reserve_bytes: u64,
        kernel: PackedInt4Kernel, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, host.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let config = *planner.config();
        if planner.group_width() != model.model_identity().group_width()
            || planner.execution_identity().task_spec != ANSWER_TASK_VERSION
            || config.planning.max_context_tokens != host.native.context_tokens {
            return Err(HostedError::ModelIdentity.into());
        }
        let chunks = planner.uses_chunk_cohorts().then_some(config.mapping.reduction.map_batch_chunks);
        let groups = planner.hierarchy_cohort_schedule().map(|schedule| schedule.max_groups);
        let quote = int4_question_document_job_requirements(planner.group_width(), config,
            request.limits, host, reduction_reserve_bytes, chunks, groups)?;
        let required = quote.document;
        if planner.max_result_bytes() != required.max_output_bytes
            || planner.max_output_tokens() != required.max_output_tokens {
            return Err(HostedError::Limits("question document job output quotation").into());
        }
        let cohort = planner.uses_native_cohorts().then(|| planner.cohort_requirements()
            .map_err(HostedError::BatchSetup)).transpose()?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(QuestionJobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        let frontier = Pending::reserve(&lease, MemoryClass::JobBuffers, required.document_workspace_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, host.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let (planner, vocabulary, request) = input.value.preparation.take()
                .ok_or(HostedError::CompletionMissing)?;
            let population = match read_population(&mut input.value.reader,
                &request, host.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            let admission = QuestionAdmission::new(&lease, model.model_identity(),
                planner.execution_identity(), config, required);
            let result = if let Some(cohort) = cohort {
                let budget = Int4SourceMapCohortBudget { context_limit: host.native.context_tokens,
                    native_memory: Int4MemoryBudget { max_kv_bytes: cohort.native.kv_bytes,
                        max_rope_bytes: cohort.native.rope_bytes,
                        max_scratch_payload_bytes: cohort.native.scratch_payload_bound },
                    max_workspace_bytes: cohort.workspace_bytes, max_result_bytes: cohort.max_result_bytes };
                let result = (|| {
                    let processor = Int4QuestionDocumentJobProcessor::new_cohorts(&planner,
                        &model.inner.loaded.value, &vocabulary, budget, kernel, admission)
                        .map_err(JobRunError::Processor)?;
                    run_population(request, &population, processor, control)
                })();
                // Discovery and hierarchy phases drain sequentially. Retain
                // the phase-peak charge without constructing a spare engine.
                drop(scratch); drop(kv);
                result
            } else {
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(&model.inner.loaded.value, host.native.context_tokens,
                        Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                            max_rope_bytes: required.native.rope_bytes,
                            max_scratch_payload_bytes: required.native.scratch_payload_bound })
                        .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
                })?;
                let result = (|| {
                    let processor = Int4QuestionDocumentJobProcessor::new(&planner,
                        &mut engine.value, &vocabulary, admission).map_err(JobRunError::Processor)?;
                    run_population(request, &population, processor, control)
                })();
                drop(engine);
                result
            };
            // Result guards survive canonical serialization, spool sync and
            // journal acknowledgement inside run_population before these drop.
            drop(population); drop(vocabulary); drop(planner); drop(input);
            drop(frontier); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}

#[cfg(test)] mod tests;
