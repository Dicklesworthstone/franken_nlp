//! Durable complete-document QA with one private original question per item.
//! Source, question, compiler and native ownership survive physical worker join.
//! Only a complete validated original-coordinate result may enter the spool.
use super::*;
use crate::{
    batch::source::quantized::question::{Int8QuestionDocumentBatchConfig,
        memory::{Int8QuestionDocumentMemoryLimits, Int8QuestionDocumentMemoryRequirement}},
    hosted::corpus::question::{QuestionAdmission, QuestionCapacity},
    jobs::{question_document::quantized::Int8QuestionDocumentJobMemoryLimits,
        runner::source::question_document::{Int8QuestionDocumentJobPlanner,
            Int8QuestionDocumentJobProcessor}},
    native_engine::strict_int8::projection::Int8ProjectionExecutor,
};

/// Simultaneous modeled payload, excluding the resident weights and any
/// independently retained caller preparation. This is not measured RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8QuestionDocumentJobRequirements {
    pub document: Int8QuestionDocumentMemoryRequirement,
    /// The common population/index/IO/journal/serialization owner, included
    /// exactly once in `document.input_bytes` together with document storage.
    pub population_buffer_bytes: u64,
    /// Charged separately by the persistent numerical-team dispatcher.
    pub team_workspace_bytes: u64,
}
impl Int8QuestionDocumentJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        let d = self.document;
        sum(&[d.input_bytes, d.native.kv_bytes, d.workspace_bytes,
            d.document_workspace_bytes, d.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Pure complete-job quote before key, artifact, population or storage IO.
/// Immutable attempts conservatively bound verification across all resumes;
/// native work and mask charges retain their existing journal counters.
#[allow(clippy::too_many_arguments)]
pub fn int8_question_document_job_requirements(config: Int8QuestionDocumentBatchConfig,
    job: JobLimits, host: JobHostLimits, input: JobInputMemory, reduction_reserve_bytes: u64,
    prefill: Option<Int8PrefillLimits>, compute_threads: Option<usize>)
    -> Result<Int8QuestionDocumentJobRequirements, HostedJobError> {
    let buffers = host.required_buffer_bytes_for_input(job, input)?;
    let required = Int8QuestionDocumentJobMemoryLimits {
        document: Int8QuestionDocumentMemoryLimits {
            context_tokens: host.native.context_tokens,
            allocator_reserve_bytes: host.native.allocator_reserve_bytes,
            preparation_reserve_bytes: host.preparation_reserve_bytes,
            io_reserve_bytes: host.io_reserve_bytes, reduction_reserve_bytes,
        },
        transport: host.transport, journal_reserve_bytes: host.journal_reserve_bytes,
        serialization_reserve_bytes: host.serialization_reserve_bytes,
    }.requirements(config, job, input).map_err(HostedError::BatchSetup)?;
    if required.population_buffer_bytes != buffers {
        return Err(HostedError::Limits("INT8 question job population accounting").into());
    }
    let mut document = required.document;
    document.workspace_bytes = crate::hosted::scored::scoring_scratch(document.workspace_bytes, prefill)?;
    let team_workspace_bytes = compute_threads.map(|width|
        dispatch::int8_prefill_team_workspace_bytes(width, prefill.map_or(1, |p| p.max_batch_rows)))
        .transpose()?.unwrap_or(0);
    let required = Int8QuestionDocumentJobRequirements {
        document, population_buffer_bytes: buffers, team_workspace_bytes,
    };
    required.total_bytes()?;
    Ok(required)
}

struct JobInput<R> {
    _retained: (Arc<SourceTaskPlanner>, Arc<ExtractionVocabulary>),
    population: Option<JobPopulation>,
    preparation: Option<(Int8QuestionDocumentJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}
/// This aggregate is owned outside the native scoped callback. Neither an
/// error nor a completed progress value drops its owners before physical join.
struct JobExecution<R> {
    input: Charged<JobInput<R>>,
    kv: Option<Pending>, scratch: Option<Pending>, frontier: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>, _frontier: Option<CommittedMemory>,
    limits: JobHostLimits,
    required: Int8QuestionDocumentJobRequirements,
    lease: EngineLease,
}

impl NlpEngine {
    /// Execute or resume complete original source/question records. The
    /// request explicitly selects original input or authenticated retained
    /// replay; failed attempts never refund native, mask or attempt authority.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_question_documents<R>(&self, model: &ResidentInt8,
        planner: Int8QuestionDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.question_document_job_scheduled(model, planner, vocabulary, request, limits,
            reduction_reserve_bytes, None, reader, cancellation)
    }

    /// One fixed team spans every pending record and the planner's optional
    /// layer-major prompt morsels. Width changes no semantic job identity.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_question_documents_with_threads<R>(&self, model: &ResidentInt8,
        planner: Int8QuestionDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        kernel: Int8Kernel, compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.question_document_job_scheduled(model, planner, vocabulary, request, limits,
            reduction_reserve_bytes, Some((kernel, compute_threads)), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn question_document_job_scheduled<R>(&self, model: &ResidentInt8,
        planner: Int8QuestionDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        team: Option<(Int8Kernel, usize)>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match team {
            None => dispatch::preflight(self, limits.native.run)?,
            Some((_, width)) => dispatch::preflight_int8_team(self, limits.native.run, width)?,
        }
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), planner.execution_identity())?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let prefill = planner.prefill_limits();
        let required = int8_question_document_job_requirements(*planner.config(), request.limits,
            limits, request.mode.input_memory(), reduction_reserve_bytes, prefill,
            team.map(|(_, width)| width))?;
        if planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)? > limits.preparation_reserve_bytes
            || planner.max_result_bytes() != required.document.max_output_bytes
            || planner.max_output_tokens() != required.document.max_output_tokens {
            return Err(HostedError::Limits("INT8 question job exact preparation and result quote").into());
        }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.document.input_bytes)?, || {
            let retained = (planner.retained_owners(), Arc::clone(&vocabulary));
            Ok(JobInput { _retained: retained, population: None,
                preparation: Some((planner, vocabulary, request)), reader })
        })?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.document.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.document.workspace_bytes)?;
        let frontier = Pending::reserve(&lease, MemoryClass::JobBuffers, required.document.document_workspace_bytes)?;
        let execution = JobExecution { input, kv: Some(kv), scratch: Some(scratch), frontier: Some(frontier),
            _kv: None, _scratch: None, _frontier: None, limits, required, lease };
        let model = model.clone();
        let result = match team {
            None => dispatch::run(self, limits.native.run, cancellation, move |control| {
                let mut execution = execution;
                execution.run(&model.inner.loaded.value, None, None, control)
            }),
            Some((kernel, width)) => dispatch::run_int8_prefill_team(self, limits.native.run,
                cancellation, model, width, kernel, prefill.map_or(1, |p| p.max_batch_rows), execution,
                move |model, execution, control, projection|
                    execution.run(model, Some(kernel), Some(projection), control)),
        }?;
        result.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> JobExecution<R> {
    fn run<'weights>(&mut self, model: &'weights CurrentCandidateInt8Model, kernel: Option<Int8Kernel>,
        projection: Option<&'weights mut (dyn Int8ProjectionExecutor + 'weights)>, control: &mut RunControl)
        -> Result<Result<JobProgress, JobRunError>, HostedError> {
        checkpoint(control)?;
        let (planner, vocabulary, request) = self.input.value.preparation.take()
            .ok_or(HostedError::CompletionMissing)?;
        let config = *planner.config();
        let identity = planner.execution_identity().clone();
        let population = match read_population(&mut self.input.value.reader, &request, self.limits.transport, control) {
            Ok(Population::Original(original)) =>
                Population::BorrowedOriginal(self.input.value.population.insert(original)),
            Ok(population) => population,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        checkpoint(control)?;
        let need = self.required.document;
        let mut engine = model.engine(self.limits.native.context_tokens, memory_budget(need.native))
            .map_err(HostedError::Model)?;
        if let Some(kernel) = kernel { engine = engine.with_kernel(kernel); }
        if let Some(projection) = projection {
            engine = engine.with_projection_executor(projection).map_err(HostedError::Native)?;
        }
        // Retained-input creation may borrow the reader. Commit only disjoint
        // fields while that authenticated population borrow remains live.
        self._kv = Some(self.kv.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        self._frontier = Some(self.frontier.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        let admission = QuestionAdmission::new(&self.lease, model.artifact_identity(), &identity,
            QuestionCapacity { config, kv_bytes: need.native.kv_bytes,
                max_result_bytes: need.max_output_bytes, max_output_tokens: need.max_output_tokens });
        let result = (|| {
            let processor = Int8QuestionDocumentJobProcessor::new(&planner, &mut engine, &vocabulary, admission)
                .map_err(JobRunError::Processor)?;
            run_population(request, &population, processor, control)
        })();
        if result.is_ok() && (engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)) {
            return Err(HostedError::Limits("INT8 question job native completion"));
        }
        drop(engine); drop(population);
        Ok(result)
    }
}

#[cfg(test)] mod tests;
