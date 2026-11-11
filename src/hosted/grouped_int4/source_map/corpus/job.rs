//! Authenticated per-record questions reuse the complete native QA processor.
use super::*;
use crate::{
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress,
        runner::{JobRunError, source::grouped_int4::question_document::{
            Int4QuestionDocumentJobPlanner, Int4QuestionDocumentJobProcessor,
            memory::{Int4QuestionDocumentJobMemoryLimits, Int4QuestionDocumentJobMemoryRequirement}}}},
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
    tasks::source_planning::grouped_int4::long::cohort::Int4SourceMapCohortRequirement,
};

struct QuestionJobInput<R> {
    preparation: Option<(Int4QuestionDocumentJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}

struct QuestionJobExecution<R> {
    input: Charged<QuestionJobInput<R>>,
    frontier: Pending,
    kv: Pending,
    scratch: Pending,
    host: JobHostLimits,
    config: Int4QuestionDocumentBatchConfig,
    required: Int4QuestionDocumentMemoryRequirement,
    cohort: Option<Int4SourceMapCohortRequirement>,
    lease: EngineLease,
}

/// Quote the same complete population, question/source frontiers, native
/// phase peak and guarded output used by the owned execution entrypoint.
#[allow(clippy::too_many_arguments)]
pub fn int4_question_document_job_requirements(width: usize,
    config: Int4QuestionDocumentBatchConfig, job: JobLimits, host: JobHostLimits,
    reduction_reserve_bytes: u64, cohort_chunks: Option<usize>, hierarchy_groups: Option<usize>,
    input: JobInputMemory)
    -> Result<Int4QuestionDocumentJobMemoryRequirement, HostedJobError> {
    let population = host.required_buffer_bytes_for_input(job, input)?;
    let required = Int4QuestionDocumentJobMemoryLimits {
        document: Int4QuestionDocumentMemoryLimits { context_tokens: host.native.context_tokens,
            allocator_reserve_bytes: host.native.allocator_reserve_bytes,
            preparation_reserve_bytes: host.preparation_reserve_bytes,
            io_reserve_bytes: host.io_reserve_bytes, reduction_reserve_bytes,
            cohort_chunks, hierarchy_groups },
        transport: host.transport, journal_reserve_bytes: host.journal_reserve_bytes,
        serialization_reserve_bytes: host.serialization_reserve_bytes,
    }.requirements(width, config, job, input).map_err(HostedError::BatchSetup)?;
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
        self.run_int4_question_document_job(model, planner, vocabulary, request, host,
            reduction_reserve_bytes, kernel, None, reader, cancellation)
    }

    /// Retain the same authenticated whole-source/question results with one
    /// persistent numerical team across pending items and all dependent
    /// synthesis/hierarchy passes. The physical width is not a recipe field;
    /// changing it on explicit resume does not renew persisted work or alter
    /// existing committed records. Native cohort schedules remain separate.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_question_documents_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4QuestionDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, host: JobHostLimits, reduction_reserve_bytes: u64,
        kernel: PackedInt4Kernel, compute_threads: usize, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_question_document_job(model, planner, vocabulary, request, host,
            reduction_reserve_bytes, kernel, Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_question_document_job<R>(&self, model: &ResidentInt4,
        planner: Int4QuestionDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, host: JobHostLimits, reduction_reserve_bytes: u64,
        kernel: PackedInt4Kernel, compute_threads: Option<usize>, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, host.native.run, width)?,
            None => dispatch::preflight(self, host.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let config = *planner.config();
        check_question_team_schedule(compute_threads, planner.uses_native_cohorts(), config)?;
        if planner.group_width() != model.model_identity().group_width()
            || planner.execution_identity().task_spec != ANSWER_TASK_VERSION
            || config.planning.max_context_tokens != host.native.context_tokens {
            return Err(HostedError::ModelIdentity.into());
        }
        check_encoding_preparation(planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)?,
            host.preparation_reserve_bytes)?;
        let chunks = planner.uses_chunk_cohorts().then_some(config.mapping.reduction.map_batch_chunks);
        let groups = planner.hierarchy_cohort_schedule().map(|schedule| schedule.max_groups);
        let quote = int4_question_document_job_requirements(planner.group_width(), config,
            request.limits, host, reduction_reserve_bytes, chunks, groups, request.mode.input_memory())?;
        let required = quote.document;
        if planner.max_result_bytes() != required.max_output_bytes
            || planner.max_output_tokens() != required.max_output_tokens {
            return Err(HostedError::Limits("question document job output quotation").into());
        }
        let cohort = planner.uses_native_cohorts().then(|| planner.cohort_requirements()
            .map_err(HostedError::BatchSetup)).transpose()?;
        let team_scratch = compute_threads.map(|threads|
            int4_task_team_workspace_bytes(threads, model.model_identity().group_width()))
            .transpose()?.unwrap_or(0);
        // The dispatcher admits/owns this extra charge once for the complete
        // invocation; it is not part of each item's durable native work.
        sum(&[required.workspace_bytes, team_scratch])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(QuestionJobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        let frontier = Pending::reserve(&lease, MemoryClass::JobBuffers, required.document_workspace_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let execution = QuestionJobExecution { input, frontier, kv, scratch, host, config,
            required, cohort, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, host.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_team(self, host.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        completed.map_err(HostedJobError::Job)
    }
}

fn check_question_team_schedule(threads: Option<usize>, cohorts: bool,
    config: Int4QuestionDocumentBatchConfig) -> Result<(), HostedError> {
    if threads.is_some() && (cohorts || config.mapping.reduction.map_batch_chunks != 1) {
        return Err(HostedError::Limits("INT4 question job team requires serial chunk and hierarchy schedules"));
    }
    Ok(())
}

impl<R: BufRead> QuestionJobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let Self { mut input, frontier, kv, scratch, host, config, required, cohort, lease } = self;
        checkpoint(control)?;
        let (planner, vocabulary, request) = input.value.preparation.take()
            .ok_or(HostedError::CompletionMissing)?;
        let population = match read_population(&mut input.value.reader,
            &request, host.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        checkpoint(control)?;
        let admission = QuestionAdmission::new(&lease, weights.identity(),
            planner.execution_identity(), config, required);
        let result = if let Some(cohort) = cohort {
            if projection.is_some() {
                return Err(HostedError::Limits("INT4 question job conflicting physical schedules"));
            }
            let budget = Int4SourceMapCohortBudget { context_limit: host.native.context_tokens,
                native_memory: Int4MemoryBudget { max_kv_bytes: cohort.native.kv_bytes,
                    max_rope_bytes: cohort.native.rope_bytes,
                    max_scratch_payload_bytes: cohort.native.scratch_payload_bound },
                max_workspace_bytes: cohort.workspace_bytes, max_result_bytes: cohort.max_result_bytes };
            let result = (|| {
                let processor = Int4QuestionDocumentJobProcessor::new_cohorts(&planner,
                    weights, &vocabulary, budget, kernel, admission).map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(scratch); drop(kv);
            result
        } else {
            let mut engine = allocate_native(kv, scratch, || {
                let engine = StrictInt4Engine::new(weights, host.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                        max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound })
                    .map_err(native_error)?.with_kernel(kernel);
                match projection {
                    Some(team) => engine.with_projection_executor(team).map_err(native_error),
                    None => Ok(engine),
                }
            })?;
            let result = (|| {
                let processor = Int4QuestionDocumentJobProcessor::new(&planner,
                    &mut engine.value, &vocabulary, admission).map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            result
        };
        // Each output guard survives serialization, spool sync and journal
        // acknowledgement; runtime cancellation/drain still outranks success.
        drop(population); drop(vocabulary); drop(planner); drop(input);
        drop(frontier); drop(lease);
        Ok(result)
    }
}

#[cfg(test)] mod tests;
