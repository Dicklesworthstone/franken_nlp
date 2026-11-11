//! Complete-document INT4 pipelines inside the existing owned-job lifecycle.
//! One original document is one durable item, including all permitted passes.
use super::*;
use super::document_native::DocumentNative;
use crate::{
    hosted::grouped_int4::source_map::{workspace::document_bytes,
        documents::admission::{check_encoding_preparation, DocumentAdmission, work_fits}},
    jobs::runner::source::grouped_int4::document::{
        Int4SourceDocumentJobPlanner, Int4SourceDocumentJobProcessor, Int4SourceDocumentTask,
    },
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
    tasks::source_planning::{SourcePlanningLimits, grouped_int4::long::{Int4SourceMapLimits,
        cohort::{Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement}}},
};
#[cfg(test)]
use crate::{grammar::mask::MaskWorkLimits,
    batch::source::grouped_int4::document::Int4SourceDocumentAdmission,
    hosted::grouped_int4::source_map::documents::admission::check_document_request};

struct DocumentJobInput<R> {
    preparation: Option<(Int4SourceDocumentJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DocumentRequirements {
    native: Int4MemoryRequirement,
    native_workspace_bytes: u64,
    input_bytes: u64,
    document_workspace_bytes: u64,
}

// One private owner crosses the blocking boundary on both schedules. Captured
// input drops before its charge and the engine lease remains owned until last.
struct DocumentExecution<R> {
    input: Charged<DocumentJobInput<R>>,
    frontier: Pending,
    kv: Pending,
    scratch: Pending,
    decoder: Option<Pending>,
    limits: JobHostLimits,
    required: DocumentRequirements,
    cohort: Option<Int4SourceMapCohortRequirement>,
    mapping: Int4SourceMapLimits,
    result_bytes: u64,
    output_tokens: u64,
    lease: EngineLease,
}

impl Int4SourceDocumentJobPlanner {
    /// Quote the maximum complete document frontier, including every allowed
    /// synthesis/hierarchy pass and independent citation-verification storage.
    /// This reuses the single-document host's checked accounting with borrowed
    /// private settings. It is a workspace quote, not a memory permit or RSS
    /// claim; the job entrypoint also validates and reserves native, input and
    /// output storage before executing any document.
    pub fn workspace_bytes(&self, _native: NativeLimits, reduction_reserve_bytes: u64)
        -> Result<u64, HostedError> {
        document_bytes(self.task(), self.task_budget(), self.document_limits(), reduction_reserve_bytes)
    }
}

impl NlpEngine {
    /// Run or resume complete original documents through the existing INT4
    /// map, keyphrase-ranking, summary-selection, cited-synthesis and
    /// hierarchical QA pipelines.
    /// Each document commits one complete result; intermediate chunks/passes
    /// are never exposed as separately resumable or successful partial items.
    ///
    /// The immutable model-bound factory and entire original population are
    /// authenticated by JobRunner before tail repair or native execution.
    /// All native work, masks, verification and reduction allowances remain
    /// nonrenewable across failures and resumes. The output guard survives
    /// spool sync and journal acknowledgement. Private questions, inputs and
    /// results are not emitted to stdout by this method.
    /// Factories configured with chunk or hierarchy cohorts execute native
    /// windows inside the same complete-document transaction;
    /// its authenticated schedule cannot be changed by resume.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source_document<R>(&self, model: &ResidentInt4,
        planner: Int4SourceDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        kernel: PackedInt4Kernel, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_document_job(model, planner, vocabulary, request, limits,
            reduction_reserve_bytes, kernel, None, reader, cancellation)
    }

    /// Execute every complete document and dependent synthesis pass on one
    /// persistent exact row team. Worker width is physical scheduling only:
    /// authenticated resume may change it without changing the fixed recipe,
    /// original item identity, work debit or committed semantic result.
    ///
    /// Team scratch is admitted in addition to the existing document/native
    /// envelope. Chunk and hierarchy-group cohorts are incompatible. Every
    /// worker joins before return, including cancellation after durable commit.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source_document_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4SourceDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        kernel: PackedInt4Kernel, compute_threads: usize, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_document_job(model, planner, vocabulary, request, limits,
            reduction_reserve_bytes, kernel, Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_source_document_job<R>(&self, model: &ResidentInt4,
        planner: Int4SourceDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        kernel: PackedInt4Kernel, compute_threads: Option<usize>, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        check_encoding_preparation(planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)?,
            limits.preparation_reserve_bytes)?;
        let mapping = planner.document_limits();
        check_team_schedule(compute_threads, planner.uses_native_cohorts(), mapping)?;
        let result_bytes = planner.max_result_bytes();
        let output_tokens = planner.output_token_ceiling().map_err(HostedError::BatchSetup)?;
        let document_workspace = planner.workspace_bytes(limits.native, reduction_reserve_bytes)?;
        let mut required = document_requirements(model.model_identity().group_width(),
            planner.task_budget(), planner.planning_limits(), mapping, result_bytes,
            request.limits, limits, document_workspace, private_question_bytes(planner.task()),
            request.mode.input_memory())?;
        let cohort = if planner.uses_native_cohorts() {
            Some(planner.cohort_requirements().map_err(HostedError::BatchSetup)?)
        } else { None };
        if let Some(quote) = cohort { required.use_cohorts(quote, limits.native.allocator_reserve_bytes)?; }
        if let Some(schedule) = planner.hierarchy_cohort_schedule() {
            required.input_bytes = sum(&[required.input_bytes,
                schedule.additional_preparation_bytes(limits.preparation_reserve_bytes)
                    .map_err(|_| HostedError::Limits("INT4 hierarchy concurrent preparation reserve"))?])?;
        }
        // The dispatcher owns the additional worker reservation. Include the
        // same pure quote before any private population or storage access.
        let team_scratch = compute_threads.map(|threads|
            int4_task_team_workspace_bytes(threads, model.model_identity().group_width()))
            .transpose()?.unwrap_or(0);
        sum(&[required.native_workspace_bytes, team_scratch])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(DocumentJobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        // The maximum frontier remains reserved even when an early document
        // is short. Later documents and hierarchy levels cannot borrow from
        // uncharged preparation or from another concurrent request.
        let frontier = Pending::reserve(&lease, MemoryClass::JobBuffers, required.document_workspace_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.native_workspace_bytes)?;
        let decoder = cohort.map(|quote| Pending::reserve(&lease, MemoryClass::JobBuffers,
            quote.workspace_bytes)).transpose()?;
        let execution = DocumentExecution { input, frontier, kv, scratch, decoder, limits,
            required, cohort, mapping, result_bytes, output_tokens, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_team(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        // A physical cancellation/drain failure takes precedence over the job
        // result. Authenticated resume reconciles a lost acknowledgement.
        completed.map_err(HostedJobError::Job)
    }
}

fn check_team_schedule(threads: Option<usize>, cohorts: bool, mapping: Int4SourceMapLimits)
    -> Result<(), HostedError> {
    if threads.is_some() && (cohorts || mapping.reduction.map_batch_chunks != 1) {
        return Err(HostedError::Limits("INT4 document job team requires serial chunk and hierarchy schedules"));
    }
    Ok(())
}

impl<R: BufRead> DocumentExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let Self { mut input, frontier, kv, scratch, decoder, limits, required, cohort,
            mapping, result_bytes, output_tokens, lease } = self;
        checkpoint(control)?;
        let (planner, vocabulary, request) = input.value.preparation.take()
            .ok_or(HostedError::CompletionMissing)?;
        let population = match read_population(&mut input.value.reader,
            &request, limits.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        checkpoint(control)?;
        let cohort = cohort.zip(decoder).map(|(quote, decoder)| (Int4SourceMapCohortBudget {
            context_limit: limits.native.context_tokens,
            native_memory: Int4MemoryBudget { max_kv_bytes: quote.native.kv_bytes,
                max_rope_bytes: quote.native.rope_bytes,
                max_scratch_payload_bytes: quote.native.scratch_payload_bound },
            max_workspace_bytes: quote.workspace_bytes, max_result_bytes: quote.max_result_bytes,
        }, decoder));
        let mut native = DocumentNative::with_projection(weights, limits.native.context_tokens,
            required.native, kernel, kv, scratch, cohort, projection)?;
        let admission = DocumentAdmission {
            lease: &lease, model: weights.identity(), base: planner.execution_identity(),
            kv_bytes: required.native.kv_bytes, mapping, result_bytes, output_tokens,
        };
        let result = (|| {
            let processor = match &mut native {
                DocumentNative::Serial(engine) => Int4SourceDocumentJobProcessor::new(
                    &planner, &mut engine.value, &vocabulary, admission),
                DocumentNative::Cohorts { budget, .. } => Int4SourceDocumentJobProcessor::new_cohorts(
                    &planner, weights, &vocabulary, *budget, kernel, admission),
            }.map_err(JobRunError::Processor)?;
            run_population(request, &population, processor, control)
        })();
        drop(native); drop(population); drop(vocabulary); drop(planner);
        drop(input); drop(frontier); drop(lease);
        Ok(result)
    }
}

impl DocumentRequirements {
    fn use_cohorts(&mut self, quote: Int4SourceMapCohortRequirement, allocator: u64)
        -> Result<(), HostedError> {
        if quote.native.kv_bytes < self.native.kv_bytes || quote.native.rope_bytes < self.native.rope_bytes
            || quote.native.scratch_payload_bound < self.native.scratch_payload_bound
            || quote.workspace_bytes == 0 || quote.max_result_bytes == 0 {
            return Err(HostedError::Limits("INT4 document job incomplete cohort declaration"));
        }
        self.native = quote.native;
        self.native_workspace_bytes = sum(&[quote.native.rope_bytes, quote.native.scratch_payload_bound, allocator])?;
        Ok(())
    }
}

fn private_question_bytes(task: &Int4SourceDocumentTask) -> u64 {
    match task {
        Int4SourceDocumentTask::Question(question) => question.question.capacity() as u64,
        Int4SourceDocumentTask::QuestionSynthesis(request)
        | Int4SourceDocumentTask::QuestionHierarchy { request, .. } =>
            request.question.question.capacity() as u64,
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn document_requirements(width: usize, budget: TaskBudget, planning: SourcePlanningLimits,
    mapping: Int4SourceMapLimits, result_bytes: u64, job: JobLimits, limits: JobHostLimits,
    document_workspace_bytes: u64, private_question_bytes: u64, input: JobInputMemory)
    -> Result<DocumentRequirements, HostedJobError> {
    budget.validate().map_err(|_| HostedError::Limits("INT4 document job task budget"))?;
    let input_bytes = sum(&[limits.required_buffer_bytes_for_input(job, input)?, private_question_bytes])?;
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width)
        .map_err(native_error)?;
    if limits.native.allocator_reserve_bytes == 0 || document_workspace_bytes == 0
        || planning.max_context_tokens != limits.native.context_tokens
        || native.kv_bytes != budget.max_kv_bytes
        || !(1..=256).contains(&mapping.chunks.max_chunks)
        || result_bytes == 0 || result_bytes > job.max_result_bytes as u64
        || !work_fits(mapping.max_model_work, job.max_work.int4_model_work())
        || mapping.max_mask_visits > job.max_work.mask_node_visits {
        return Err(HostedError::Limits("INT4 document job complete resident and lifetime envelope").into());
    }
    let native_workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    Ok(DocumentRequirements { native, native_workspace_bytes, input_bytes, document_workspace_bytes })
}

#[cfg(test)]
mod tests;
