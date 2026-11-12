//! Authenticated durable audits reuse the complete-source corpus processor.
use super::*;
use std::io::BufRead;
use super::corpus::{DocumentAdmission, DocumentCapacity, document_output_bytes};
use crate::{
    batch::judge::grouped_int4::FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES,
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork,
        runner::{JobRunError, scored::grouped_int4::faithfulness_document::{
            Int4FaithfulnessDocumentJobPlanner, Int4FaithfulnessDocumentJobProcessor}}},
    native_engine::strict_int4::{projection::Int4ProjectionExecutor, weights::Int4ModelWeights},
};

struct JobInput<R> {
    preparation: Option<(Int4FaithfulnessDocumentJobPlanner, SourceJobRequest)>,
    reader: R,
}

struct JobExecution<R> {
    input: Charged<JobInput<R>>,
    kv: Pending,
    scratch: Pending,
    limits: JobHostLimits,
    required: Int4FaithfulnessMapMemoryRequirement,
    capacity: DocumentCapacity,
    lease: EngineLease,
}

impl NlpEngine {
    /// Start or resume complete-source judgments with fixed private policy
    /// and limits. The exact original source and claim are authenticated per
    /// item. Every chunk and witness must finish before the complete result
    /// enters the spool-first, journal-second commit path.
    ///
    /// Model, recipe and whole population authenticate before tail repair or
    /// inference. All five native work debits survive failure and restart. The
    /// reader, population, planner, native state and guarded result remain in
    /// the physical invocation until serialization and persistence drain.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_faithfulness_document<R>(&self, model: &ResidentInt4,
        planner: Int4FaithfulnessDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_faithfulness_document_job(model, planner, request, limits, kernel, None,
            reader, cancellation)
    }

    /// Reuse one persistent numerical row team through every source, claim,
    /// chunk and evidence head in this durable invocation. Physical width
    /// does not change policy, authenticated recipes or lifetime work debits.
    /// Worker scratch is admitted before population IO and all workers join
    /// before return. Explicit native chunk cohorts cannot be combined.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_faithfulness_document_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4FaithfulnessDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: usize, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_faithfulness_document_job(model, planner, request, limits, kernel, Some(compute_threads),
            reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_faithfulness_document_job<R>(&self, model: &ResidentInt4,
        planner: Int4FaithfulnessDocumentJobPlanner, request: SourceJobRequest,
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
        judge_task::check_model(planner.execution_identity(), model.model_identity()).map_err(judge_error)?;
        if planner.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let mapping = planner.document_limits();
        let cohort_chunks = planner.cohort_chunks();
        check_team_schedule(compute_threads, cohort_chunks)?;
        let required = requirements(planner.group_width(), planner.task_budget(), planner.planning_limits(),
            mapping, request.limits, limits, planner.max_heads(), cohort_chunks, request.mode.input_memory())?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, planner.group_width())).transpose()?.unwrap_or(0);
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
            required.output_reservation_bytes, team_scratch])?;
        if planner.max_result_bytes() != document_output_bytes(mapping, cohort_chunks)? {
            return Err(HostedError::Limits("faithfulness document job result envelope").into());
        }
        let capacity = DocumentCapacity { context: limits.native.context_tokens,
            kv_bytes: required.native.kv_bytes, mapping, cohort_chunks, max_heads: planner.max_heads(),
            max_result_bytes: planner.max_result_bytes(), task_budget: planner.task_budget() };
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
        // Physical drain and terminal cancellation take precedence over an
        // individual durable acknowledgement. Explicit resume reconciles it.
        completed.map_err(HostedJobError::Job)
    }
}

fn check_team_schedule(threads: Option<usize>, cohort_chunks: Option<usize>) -> Result<(), HostedError> {
    if threads.is_some() && cohort_chunks.is_some() {
        return Err(HostedError::Limits("INT4 faithfulness document job team cannot combine with native chunk cohorts"));
    }
    Ok(())
}

impl<R: BufRead> JobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        // Capture the aggregate owner so an already-cancelled queued task
        // cannot outlive its reader, private recipe or resource charge.
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
            let kv = kv.commit()?; let scratch = scratch.commit()?;
            let result = (|| {
                let processor = Int4FaithfulnessDocumentJobProcessor::new_cohorts(planner,
                    weights, kernel, map_memory(limits), admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(scratch); drop(kv);
            result
        } else {
            let mut engine = allocate_native(kv, scratch, || {
                let engine = StrictInt4Engine::new(weights, limits.native.context_tokens,
                    native_budget(required.native)).map_err(native_error)?.with_kernel(kernel);
                match projection {
                    Some(team) => engine.with_projection_executor(team).map_err(native_error),
                    None => Ok(engine),
                }
            })?;
            let result = (|| {
                let processor = Int4FaithfulnessDocumentJobProcessor::new(planner, &mut engine.value, admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            result
        };
        // Complete guarded outputs drop inside run_population before the
        // population reader and its journal/serialization reservation.
        drop(population); drop(input); drop(lease);
        Ok(result)
    }
}

fn map_memory(limits: JobHostLimits) -> Int4FaithfulnessMapMemoryLimits {
    Int4FaithfulnessMapMemoryLimits { context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes }
}

/// The authenticated original population coexists with one complete map
/// frontier and the frozen identities retained during exact recompilation.
#[allow(clippy::too_many_arguments)]
fn requirements(width: usize, task: TaskBudget, planning: JudgeLimits,
    mapping: Int4FaithfulnessMapLimits, job: JobLimits, host: JobHostLimits,
    max_heads: usize, cohort_chunks: Option<usize>, input: JobInputMemory) -> Result<Int4FaithfulnessMapMemoryRequirement, HostedJobError> {
    let population = host.required_buffer_bytes_for_input(job, input)?;
    crate::batch::judge::grouped_int4::validate_limits(task, planning, mapping,
        job.max_work.int4_model_work()).map_err(HostedError::BatchSetup)?;
    if host.native.context_tokens != mapping.chunks.context_tokens
        || mapping.chunks.max_input_bytes > job.max_input_bytes_per_item
        || job.max_work.mask_node_visits != 0
        || !JobWork::from_int4(mapping.max_model_work, 0).fits(job.max_work)
        || document_output_bytes(mapping, cohort_chunks)? > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("faithfulness complete-document durable envelope").into());
    }
    let memory = map_memory(host);
    let mut required = match cohort_chunks {
        None => memory.requirements_with_max_heads(width, task, planning, mapping, max_heads),
        Some(chunks) => memory.cohort_requirements(width, task, planning, mapping,
            max_heads, chunks).map(|required| required.memory),
    }.map_err(map_error)?;
    let frontier = required.input_bytes.checked_sub(host.preparation_reserve_bytes)
        .ok_or(HostedError::Limits("faithfulness document preparation arithmetic"))?;
    let identities = (mapping.chunks.max_chunks as u64).checked_mul(FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("faithfulness document identity arithmetic"))?;
    // Preparation is shared by both quotes; population and map source/claim
    // are separate owners and must remain charged together.
    required.input_bytes = sum(&[population, frontier, identities])?;
    sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
        required.output_reservation_bytes])?;
    Ok(required)
}

#[cfg(test)] mod tests;
