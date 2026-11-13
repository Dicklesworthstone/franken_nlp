//! Owned resumable classification, sentiment and judgment on resident INT4.
//! The existing job runner owns authenticated replay, durable work debits and
//! spool-first publication; this host owns all physical execution resources.
use super::*;
use std::io::BufRead;
use crate::{
    batch::scored_int4::Int4ScoredBatchLimits,
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress,
        runner::{JobRunError, scored::grouped_int4::{Int4ScoredJobPlanner,
            Int4ScoredJobProcessor, Int4ScoredJobTask}}},
    native_engine::{constrained_int4, strict_int4::projection::Int4ProjectionExecutor},
    tasks::ir::TaskBudget,
};
use super::scored_batch::ScoredAdmission;
mod document;

struct JobInput<P: Int4ScoredJobTask, R> {
    preparation: Option<(Int4ScoredJobPlanner<P>, SourceJobRequest)>,
    reader: R,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Requirements { native: Int4MemoryRequirement, workspace_bytes: u64, input_bytes: u64 }
impl Requirements {
    fn with_encoding_workspace(mut self, workspace: u64) -> Result<Self, HostedError> {
        self.input_bytes = sum(&[self.input_bytes, workspace])?;
        sum(&[self.input_bytes, self.native.kv_bytes, self.workspace_bytes])?;
        Ok(self)
    }
}

struct ScoredJobExecution<P: Int4ScoredJobTask, R> {
    input: Charged<JobInput<P, R>>, kv: Pending, scratch: Pending,
    limits: JobHostLimits, required: Requirements, lease: EngineLease,
}

impl NlpEngine {
    /// Start or resume one fixed scoring task over the complete original
    /// population. P is sealed to the real model-bound classification,
    /// sentiment and judge planners; callers supply no native admission hook.
    ///
    /// The frozen private recipe and exact original population authenticate
    /// before tail repair or native work. Committed items are skipped and all
    /// five native work counters remain charged across failed attempts and
    /// resumes. Every stored item contains its complete multihead task result.
    ///
    /// Supplying SourceJobRequest explicitly authorizes result retention in
    /// the existing owner-only job protocol. Its explicit input-retention mode
    /// controls the input spool; ordinary create/resume keeps external input.
    /// This method never generates keys, retries automatically or emits results
    /// to stdout.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_scored<P, R>(&self, model: &ResidentInt4,
        planner: Int4ScoredJobPlanner<P>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where P: Int4ScoredJobTask, R: BufRead + Send + 'static {
        self.run_int4_scored_job(model, planner, request, limits, kernel, None, reader, cancellation)
    }

    /// Keep one fixed numerical team across authenticated population loading,
    /// every scored head and durable attempt, commit and optional materialization.
    /// Worker width is physical metadata, so explicit resume may change it
    /// without changing recipes or refunding any work. Widths 2..=64 require
    /// admitted worker scratch and the installed process thread envelope.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_scored_with_threads<P, R>(&self, model: &ResidentInt4,
        planner: Int4ScoredJobPlanner<P>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: usize, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where P: Int4ScoredJobTask, R: BufRead + Send + 'static {
        self.run_int4_scored_job(model, planner, request, limits, kernel, Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_scored_job<P, R>(&self, model: &ResidentInt4,
        planner: Int4ScoredJobPlanner<P>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: Option<usize>, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where P: Int4ScoredJobTask, R: BufRead + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.execution_identity().task_spec != P::TASK {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let work = planner.native_limits();
        let required = requirements(model.model_identity().group_width(), planner.task_ceiling(),
            work, request.limits, limits, request.mode.input_memory())?
            .with_encoding_workspace(planner.encoding_workspace_bytes())?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, model.model_identity().group_width())).transpose()?.unwrap_or(0);
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes, team_scratch])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(JobInput { preparation: Some((planner, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let execution = ScoredJobExecution { input, kv, scratch, limits, required, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_team(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        // Physical drain/cancellation failure takes precedence over a job
        // result. Durable progress is reconciled by authenticated resume.
        completed.map_err(HostedJobError::Job)
    }
}

impl<P: Int4ScoredJobTask, R: BufRead> ScoredJobExecution<P, R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        // Keep the transferred aggregate whole: its reader, private settings
        // and key drop before their charge, including queued cancellation.
        let Self { mut input, kv, scratch, limits, required, lease } = self;
        checkpoint(control)?;
        let (planner, request) = input.value.preparation.take()
            .ok_or(HostedError::CompletionMissing)?;
        let work = planner.native_limits();
        let population = match read_population(&mut input.value.reader,
            &request, limits.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        checkpoint(control)?;
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
        let admission = ScoredAdmission::for_job(&lease, weights.identity(),
            limits.native.context_tokens, required.native.kv_bytes,
            work.max_item_work, P::TASK, request.limits.max_result_bytes as u64);
        let result = (|| {
            let processor = Int4ScoredJobProcessor::new(planner, &mut engine.value, admission)
                .map_err(JobRunError::Processor)?;
            // The runner authenticates the frozen recipe before recovery,
            // persists work before execution and holds each output guard
            // through spool sync and journal acknowledgement.
            run_population(request, &population, processor, control)
        })();
        drop(engine);
        drop(population);
        drop(input);
        drop(lease);
        Ok(result)
    }
}

fn requirements(width: usize, task: TaskBudget, work: Int4ScoredBatchLimits,
    job: JobLimits, limits: JobHostLimits, input: JobInputMemory) -> Result<Requirements, HostedJobError> {
    task.validate().map_err(|_| HostedError::Limits("INT4 scored job task ceiling"))?;
    work.validate().map_err(HostedError::BatchSetup)?;
    let input_bytes = limits.required_buffer_bytes_for_input(job, input)?;
    if job.max_work != crate::jobs::JobWork::from_int4(work.max_model_work, 0) {
        return Err(HostedError::Limits("INT4 scored job lifetime work").into());
    }
    if limits.native.allocator_reserve_bytes == 0 {
        return Err(HostedError::Limits("INT4 scored job allocator reserve").into());
    }
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width)
        .map_err(native_error)?;
    if native.kv_bytes == 0 || native.kv_bytes > task.max_kv_bytes
        || task.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("INT4 scored job full KV and stored-result envelope").into());
    }
    let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    Ok(Requirements { native, workspace_bytes, input_bytes })
}

#[cfg(test)]
mod tests;
