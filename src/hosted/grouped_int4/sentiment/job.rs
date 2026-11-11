//! Complete sentiment documents on the authenticated owned-job boundary.
//! The live corpus and durable route share the same native processor and
//! admission checks; a complete original document remains the commit unit.
use super::*;
use std::io::BufRead;
use super::corpus::{DocumentAdmission, DocumentCapacity, document_output_bytes};
use crate::native_engine::strict_int4::projection::Int4ProjectionExecutor;
use crate::{
    batch::sentiment::grouped_int4::SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES,
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork,
        runner::{JobRunError, scored::grouped_int4::sentiment_document::{
            Int4SentimentDocumentJobPlanner, Int4SentimentDocumentJobProcessor}}},
    native_engine::{constrained_int4, strict_int4::cohort::MAX_INT4_SEQUENCES},
    tasks::sentiment::{SentimentLimits, grouped_int4::long::{Int4SentimentMapError,
        Int4SentimentMapLimits, memory::{Int4SentimentMapMemoryLimits, Int4SentimentMapMemoryRequirement}}},
};

struct JobInput<R> {
    preparation: Option<(Int4SentimentDocumentJobPlanner, SourceJobRequest)>,
    reader: R,
}

struct JobExecution<R> {
    input: Charged<JobInput<R>>,
    kv: Pending,
    scratch: Pending,
    limits: JobHostLimits,
    required: Int4SentimentMapMemoryRequirement,
    capacity: DocumentCapacity,
    lease: EngineLease,
}

impl NlpEngine {
    /// Start or resume complete-document sentiment with fixed private axes
    /// and scoring policy. Every chunk and every dimension must finish before
    /// a document result enters the spool-first, journal-second commit path.
    ///
    /// The original population and immutable model/recipe/limits authenticate
    /// before tail repair or inference. All five native work debits survive
    /// failures and restart. The supplied reader, planner, native state and
    /// result reservations remain owned until physical completion.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sentiment_document<R>(&self, model: &ResidentInt4,
        planner: Int4SentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_sentiment_document_job(model, planner, request, limits, kernel, None,
            reader, cancellation)
    }

    /// Keep one persistent numerical row team for the entire durable
    /// invocation. Physical width does not change fixed axes, thresholds,
    /// score space, compact-input policy, authenticated recipe or work debits.
    /// Worker scratch is reserved before population IO; every worker joins
    /// before return. Explicit native chunk cohorts cannot be combined.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sentiment_document_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4SentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: usize, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_sentiment_document_job(model, planner, request, limits, kernel, Some(compute_threads),
            reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_sentiment_document_job<R>(&self, model: &ResidentInt4,
        planner: Int4SentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: Option<usize>, reader: R,
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
        if planner.group_width() != model.model_identity().group_width()
            || planner.execution_identity().task_spec != "sentiment-v1" {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let mapping = planner.document_limits();
        let max_axes = planner.max_axes();
        let cohort_chunks = planner.cohort_chunks();
        check_team_schedule(compute_threads, cohort_chunks)?;
        let required = requirements(planner.group_width(), planner.task_budget(),
            planner.planning_limits(), mapping, request.limits, limits, max_axes, cohort_chunks,
            request.mode.input_memory())?
            .with_input_encoding(planner.document_encoding()).map_err(map_error)?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, planner.group_width())).transpose()?.unwrap_or(0);
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
            required.output_reservation_bytes, team_scratch])?;
        if planner.max_result_bytes() != document_output_bytes(mapping, cohort_chunks)?
            || planner.retained_defaults_bytes().map_err(HostedError::BatchSetup)?
                > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("sentiment document job preparation or result envelope").into());
        }
        let capacity = DocumentCapacity { context: limits.native.context_tokens,
            kv_bytes: required.native.kv_bytes, mapping, max_axes, cohort_chunks,
            max_result_bytes: planner.max_result_bytes() };
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(JobInput { preparation: Some((planner, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let execution = JobExecution { input, kv, scratch, limits, required, capacity, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_team(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        // A durable acknowledgement never replaces physical worker drain or
        // terminal cancellation. Authenticated resume reconciles any commit.
        completed.map_err(HostedJobError::Job)
    }
}

fn check_team_schedule(threads: Option<usize>, cohort_chunks: Option<usize>) -> Result<(), HostedError> {
    if threads.is_some() && cohort_chunks.is_some() {
        return Err(HostedError::Limits("INT4 sentiment document job team cannot combine with native chunk cohorts"));
    }
    Ok(())
}

impl<R: BufRead> JobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        // Capture the aggregate owner, including queued reader/key/defaults,
        // so cancellation cannot release its memory charge prematurely.
        let Self { mut input, kv, scratch, limits, required, capacity, lease } = self;
        checkpoint(control)?;
        let (planner, request) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
        let population = match read_population(&mut input.value.reader,
            &request, limits.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        checkpoint(control)?;
        let admission = DocumentAdmission { lease: &lease, model: weights.identity(),
            capacity, output_bytes: required.output_reservation_bytes };
        let result = if capacity.cohort_chunks.is_some() {
            // Each exact ragged window is created and destroyed by the
            // existing map executor while the declared peak stays charged.
            let kv = kv.commit()?;
            let scratch = scratch.commit()?;
            let result = (|| {
                let processor = Int4SentimentDocumentJobProcessor::new_cohorts(planner,
                    weights, kernel, map_memory(limits, capacity.max_axes, capacity.cohort_chunks), admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(scratch);
            drop(kv);
            result
        } else {
            let mut engine = allocate_native(kv, scratch, || {
                let engine = StrictInt4Engine::new(weights, limits.native.context_tokens,
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
                let processor = Int4SentimentDocumentJobProcessor::new(planner, &mut engine.value, admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            result
        };
        // The job runner and result guards drop before the original
        // population and its input, journal and serialization allowance.
        drop(population);
        drop(input);
        drop(lease);
        Ok(result)
    }
}

fn map_memory(limits: JobHostLimits, max_axes: usize, cohort_chunks: Option<usize>)
    -> Int4SentimentMapMemoryLimits {
    Int4SentimentMapMemoryLimits { context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes, max_axes, cohort_chunks }
}

/// A pure ownership quote: the whole population coexists with one complete
/// map frontier and the frozen identities retained during exact recompilation.
#[allow(clippy::too_many_arguments)]
fn requirements(width: usize, task: TaskBudget, planning: SentimentLimits,
    mapping: Int4SentimentMapLimits, job: JobLimits, host: JobHostLimits,
    max_axes: usize, cohort_chunks: Option<usize>, input: JobInputMemory) -> Result<Int4SentimentMapMemoryRequirement, HostedJobError> {
    let population = host.required_buffer_bytes_for_input(job, input)?;
    crate::batch::sentiment::grouped_int4::validate_limits(task, planning, mapping,
        job.max_work.int4_model_work()).map_err(HostedError::BatchSetup)?;
    if host.native.context_tokens != mapping.chunks.context_tokens
        || !(1..=4).contains(&max_axes)
        || cohort_chunks.is_some_and(|chunks| chunks == 0 || chunks > mapping.chunks.max_chunks
            || chunks.checked_mul(max_axes).is_none_or(|heads| heads > MAX_INT4_SEQUENCES))
        || job.max_work.mask_node_visits != 0
        || !JobWork::from_int4(mapping.max_model_work, 0).fits(job.max_work)
        || document_output_bytes(mapping, cohort_chunks)? > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("sentiment complete-document durable envelope").into());
    }
    let mut required = map_memory(host, max_axes, cohort_chunks)
        .requirements(width, task, planning, mapping).map_err(map_error)?;
    let frontier = required.input_bytes.checked_sub(host.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("sentiment document preparation arithmetic"))?;
    let identities = (mapping.chunks.max_chunks as u64).checked_mul(SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("sentiment document identity arithmetic"))?;
    // Preparation is shared by both quotes; input population and map source
    // are not shared and must remain charged together.
    required.input_bytes = sum(&[population, frontier, identities])?;
    sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
        required.output_reservation_bytes])?;
    Ok(required)
}

fn map_error(error: Int4SentimentMapError) -> HostedError {
    HostedError::Int4(Int4HostError::SentimentMap(error))
}

#[cfg(test)]
mod tests;
