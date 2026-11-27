//! Authenticated complete-document sampled jobs on one resident INT8 engine.
//! Original input, retained replay, durable work and result publication use
//! the existing owned-job contract; only complete validated maps may commit.
use super::*;
use std::io::BufRead;
use crate::{
    batch::sentiment::sampled::{SAMPLED_SENTIMENT_JOB_RECIPE_BYTES,
        quantized::Int8SampledSentimentBatchLimits},
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest,
        Population, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork, population::JobPopulation,
        runner::{JobRunError, sampled_sentiment::quantized::long::{Int8SampledSentimentDocumentJobPlanner,
            Int8SampledSentimentDocumentJobProcessor}}},
    native_engine::strict_int8::projection::Int8ProjectionExecutor,
    tasks::sentiment::sampled::quantized::long::memory::MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
};
use super::corpus::DocumentAdmission;

/// Simultaneous admitted payload, excluding resident weights and any
/// independently retained caller preparation. This is not measured RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SampledSentimentDocumentJobRequirements {
    /// Complete map preparation, native state, sampler and guarded result.
    /// Input also includes population/index, IO, journal and serializer storage.
    pub document: Int8SampledSentimentMapMemoryRequirement,
    /// Reserved separately by the persistent numerical-team dispatcher.
    pub team_workspace_bytes: u64,
}
impl Int8SampledSentimentDocumentJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        let d = self.document;
        sum(&[d.input_bytes, d.native.kv_bytes, d.workspace_bytes,
            d.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Pure whole-job admission before private source, key, storage or model IO.
/// Per-chunk task output and complete-document output are separate ceilings.
#[allow(clippy::too_many_arguments)]
pub fn int8_sampled_sentiment_document_job_requirements(task: TaskBudget,
    planning: SampledSentimentLimits, mapping: Int8SampledSentimentMapLimits,
    options: &SampledSentimentOptions, encoding: &UntrustedEncoding,
    work: Int8SampledSentimentBatchLimits, job: JobLimits, host: JobHostLimits,
    input: JobInputMemory, prefill: Option<Int8PrefillLimits>, compute_threads: Option<usize>)
    -> Result<Int8SampledSentimentDocumentJobRequirements, HostedJobError> {
    work.validate().map_err(HostedError::BatchSetup)?;
    let buffers = host.required_buffer_bytes_for_input(job, input)?;
    let lifetime = JobWork { model: work.max_model_work, mask_node_visits: work.masks.max_visits_per_run };
    if job.max_work != lifetime || job.max_id_bytes > 256
        || mapping.max_model_work != work.max_item_work
        || mapping.chunks.max_input_bytes > job.max_input_bytes_per_item
        || mapping.max_result_bytes > job.max_result_bytes as u64
        || planning.max_input_bytes > job.max_input_bytes_per_item
        || task.max_output_bytes.min(planning.max_result_bytes) > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("INT8 sampled document job exact lifetime and document ceilings").into());
    }
    let mut document = Int8SampledSentimentMapMemoryLimits {
        context_tokens: host.native.context_tokens,
        preparation_reserve_bytes: MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: host.native.allocator_reserve_bytes,
        max_axes: SentimentAxis::ALL.len(), max_sampler_bytes: work.max_sampler_bytes,
    }.requirements(task, planning, mapping, options, encoding).map_err(HostedError::SampledSentimentMap)?;
    let preparation = document.input_bytes.checked_add(SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64)
        .ok_or(HostedError::Limits("INT8 sampled document job recipe arithmetic"))?;
    if preparation > host.preparation_reserve_bytes {
        return Err(HostedError::Limits("INT8 sampled document job complete preparation reservation").into());
    }
    // The host reserve already prices the entire compiler/map/recipe peak.
    document.input_bytes = buffers;
    document.workspace_bytes = crate::hosted::scored::scoring_scratch(document.workspace_bytes, prefill)?;
    let rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    let team_workspace_bytes = compute_threads.map(|width|
        dispatch::int8_prefill_team_workspace_bytes(width, rows)).transpose()?.unwrap_or(0);
    let required = Int8SampledSentimentDocumentJobRequirements { document, team_workspace_bytes };
    required.total_bytes()?;
    Ok(required)
}

struct JobInput<R> {
    _retained: (Arc<Int8SampledSentimentPlanner>, Arc<SampledSentimentVocabulary>),
    population: Option<JobPopulation>,
    preparation: Option<(Int8SampledSentimentDocumentJobPlanner, SourceJobRequest)>,
    reader: R,
}
/// Admission remains outside the scoped native callback through success,
/// cancellation and unwind. A returned progress value waits for worker join.
struct JobExecution<R> {
    input: Charged<JobInput<R>>,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    limits: JobHostLimits,
    required: Int8SampledSentimentDocumentJobRequirements,
    lease: EngineLease,
}

impl NlpEngine {
    /// Start or resume independent sampled maps over complete original
    /// documents. SourceJobRequest explicitly chooses original input or
    /// authenticated retained replay. Failed attempts never refund work.
    pub fn job_int8_sampled_sentiment_documents<R>(&self, model: &ResidentInt8,
        planner: Int8SampledSentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int8_sampled_sentiment_document_job(model, planner, request, limits,
            None, reader, cancellation)
    }

    /// One fixed team spans the population and the planner's optional prompt
    /// morsels. Input, compiler, vocabulary and charges remain owned through
    /// physical worker completion and any final result materialization.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sampled_sentiment_documents_with_threads<R>(&self, model: &ResidentInt8,
        planner: Int8SampledSentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: Int8Kernel, compute_threads: usize,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int8_sampled_sentiment_document_job(model, planner, request, limits,
            Some((kernel, compute_threads)), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int8_sampled_sentiment_document_job<R>(&self, model: &ResidentInt8,
        planner: Int8SampledSentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, team: Option<(Int8Kernel, usize)>, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match team {
            None => dispatch::preflight(self, limits.native.run)?,
            Some((_, width)) => dispatch::preflight_int8_team(self, limits.native.run, width)?,
        }
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), planner.base_execution_identity())?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let prefill = planner.prefill_limits();
        let required = int8_sampled_sentiment_document_job_requirements(planner.task_budget(),
            planner.planning_limits(), planner.mapping_limits(), planner.sampling_options(),
            planner.document_encoding(), planner.native_limits(), request.limits, limits,
            request.mode.input_memory(), prefill, team.map(|(_, width)| width))?;
        if planner.preparation_requirement_bytes() > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("INT8 sampled document job retained compiler").into());
        }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.document.input_bytes)?, || {
            let retained = planner.retained_owners();
            Ok(JobInput { _retained: retained, population: None,
                preparation: Some((planner, request)), reader })
        })?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.document.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.document.workspace_bytes)?;
        let execution = JobExecution { input, kv: Some(kv), scratch: Some(scratch), _kv: None, _scratch: None,
            limits, required, lease };
        let model = model.clone();
        let result = match team {
            None => dispatch::run(self, limits.native.run, cancellation, move |control| {
                let mut execution = execution;
                execution.run(&model.inner.loaded.value, None, None, control)
            }),
            Some((kernel, width)) => dispatch::run_int8_prefill_team(self, limits.native.run,
                cancellation, model, width, kernel, prefill.map_or(1, |limits| limits.max_batch_rows),
                execution, move |model, execution, control, projection|
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
        let (planner, request) = self.input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
        let mapping = planner.mapping_limits(); let work = planner.native_limits();
        let options = planner.sampling_options().clone();
        let population = match read_population(&mut self.input.value.reader, &request, self.limits.transport, control) {
            Ok(Population::Original(original)) => {
                // Keep these same authenticated bytes outside the runner and
                // its borrowed processor until every native child has joined.
                Population::BorrowedOriginal(self.input.value.population.insert(original))
            }
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
        // Retained-input creation may borrow the reader. Mutate only disjoint
        // claim fields while that population borrow remains live.
        self._kv = Some(self.kv.take().ok_or(HostedError::Limits("INT8 sampled job KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take()
            .ok_or(HostedError::Limits("INT8 sampled job scratch admission consumed"))?.commit()?);
        let admission = DocumentAdmission::for_job(&self.lease, model.artifact_identity(),
            self.limits.native.context_tokens, need.native.kv_bytes, mapping, work,
            self.limits.preparation_reserve_bytes, need.output_reservation_bytes,
            need.max_output_bytes, need.max_output_tokens, &options);
        let result = (|| {
            let processor = Int8SampledSentimentDocumentJobProcessor::new(planner, &mut engine, admission)
                .map_err(JobRunError::Processor)?;
            run_population(request, &population, processor, control)
        })();
        if result.is_ok() && (engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)) {
            return Err(HostedError::Limits("INT8 sampled document job native completion"));
        }
        drop(engine); drop(population);
        Ok(result)
    }
}

#[cfg(test)] mod tests;
