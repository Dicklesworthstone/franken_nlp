//! Owned resumable sampled-sentiment bundles on one resident native model.
//! The existing job protocol authenticates the complete population and frozen
//! private sampler recipe, debits work durably and publishes whole bundles.
//! This host retains native, input and output ownership through physical drain.

use super::*;
use std::io::BufRead;
use crate::{
    batch::sentiment::sampled::SampledSentimentBatchLimits,
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork,
        runner::{JobRunError, sampled_sentiment::{SampledSentimentJobPlanner, SampledSentimentJobProcessor}}},
    native_engine::{constrained_int4, strict_int4::projection::Int4ProjectionExecutor},
};
use super::corpus::Admission;

struct JobInput<R> {
    preparation: Option<(SampledSentimentJobPlanner, SourceJobRequest)>,
    reader: R,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Requirements { native: Int4MemoryRequirement, workspace_bytes: u64, input_bytes: u64 }

struct JobExecution<R> {
    input: Charged<JobInput<R>>, kv: Pending, scratch: Pending,
    limits: JobHostLimits, required: Requirements, prefill: Option<Int4PrefillLimits>, lease: EngineLease,
}

impl NlpEngine {
    /// Start or resume a fixed independent-sampling policy over authenticated
    /// population records. Each committed result contains every requested
    /// dimension and sample; incomplete bundles never become successful items.
    ///
    /// SourceJobRequest explicitly authorizes result retention. Its input
    /// retention mode separately controls whether original documents are saved.
    /// The caller supplies the protected key and ID; this method never invents
    /// a seed, retries, falls back to another model, or writes results to stdout.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sampled_sentiment<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_sampled_sentiment_job(model, planner, request, limits, kernel,
            None, None, reader, cancellation)
    }

    /// Use one admitted causal prompt workspace across every fresh sample.
    /// Physical morsel width does not change the durable recipe, item address
    /// or lifetime work allowance; resume can select another legal schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sampled_sentiment_layer_major<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, prefill: Int4PrefillLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_sampled_sentiment_job(model, planner, request, limits, kernel,
            None, Some(prefill), reader, cancellation)
    }

    /// Keep one fixed numerical projection team across population loading,
    /// every sample and durable commit, and optional ordered materialization.
    /// All workers join before progress is returned. A terminal runtime or
    /// physical-drain failure takes precedence over committed progress; exact
    /// authenticated resume reconciles the journal without refunding work.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sampled_sentiment_with_threads<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_sampled_sentiment_job(model, planner, request, limits, kernel,
            Some(compute_threads), None, reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_sampled_sentiment_job<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: Option<usize>,
        prefill: Option<Int4PrefillLimits>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        check_schedule(compute_threads, prefill)?;
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
            || planner.execution_identity().task_spec != "sentiment-v1"
            || planner.execution_identity().sampler_version != constrained_int4::JSON_SAMPLER_VERSION {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let result_bytes = planner.task_budget().max_output_bytes.min(planner.planning_limits().max_result_bytes);
        let required = requirements(planner.group_width(), planner.task_budget(), planner.native_limits(),
            planner.preparation_requirement_bytes(), result_bytes, request.limits, limits,
            request.mode.input_memory(), prefill)?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, model.model_identity().group_width()))
            .transpose()?.unwrap_or(0);
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes, team_scratch])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(JobInput { preparation: Some((planner, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let execution = JobExecution { input, kv, scratch, limits, required, prefill, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_team(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        completed.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> JobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        // Whole-aggregate transfer ensures queued cancellation drops private
        // planner/key/reader storage before their reservation disappears.
        let Self { mut input, kv, scratch, limits, required, prefill, lease } = self;
        checkpoint(control)?;
        let (planner, request) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
        let work = planner.native_limits();
        let population = match read_population(&mut input.value.reader, &request, limits.transport, control) {
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
        let admission = Admission::for_job(&lease, weights.identity(), limits.native.context_tokens,
            required.native.kv_bytes, work, limits.preparation_reserve_bytes,
            request.limits.max_result_bytes as u64);
        let result = (|| {
            let processor = match prefill {
                None => SampledSentimentJobProcessor::new(planner, &mut engine.value, admission),
                Some(prefill) => SampledSentimentJobProcessor::new_layer_major(
                    planner, &mut engine.value, admission, prefill),
            }.map_err(JobRunError::Processor)?;
            // The runner authenticates before recovery, durably charges the
            // entire attempted bundle, and retains each output guard through
            // spool sync and the journal acknowledgement.
            run_population(request, &population, processor, control)
        })();
        drop(engine);
        drop(population);
        drop(input);
        drop(lease);
        Ok(result)
    }
}

fn check_schedule(threads: Option<usize>, prefill: Option<Int4PrefillLimits>) -> Result<(), HostedError> {
    if threads.is_some() && prefill.is_some() {
        return Err(HostedError::Limits("sampled jobs use separate prefill and compute-team schedules"));
    }
    if let Some(limits) = prefill { limits.validate().map_err(native_error)?; }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn requirements(width: usize, task: TaskBudget, work: SampledSentimentBatchLimits,
    preparation_bytes: u64, result_bytes: u64, job: JobLimits, limits: JobHostLimits,
    input: JobInputMemory, prefill: Option<Int4PrefillLimits>) -> Result<Requirements, HostedJobError> {
    task.validate().map_err(|_| HostedError::Limits("sampled sentiment job task ceiling"))?;
    work.validate().map_err(HostedError::BatchSetup)?;
    let input_bytes = limits.required_buffer_bytes_for_input(job, input)?;
    if job.max_work != JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run) {
        return Err(HostedError::Limits("sampled sentiment job lifetime native and mask work").into());
    }
    if preparation_bytes == 0 || preparation_bytes > limits.preparation_reserve_bytes
        || limits.native.allocator_reserve_bytes == 0 {
        return Err(HostedError::Limits("sampled sentiment job preparation or allocator reserve").into());
    }
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    if native.kv_bytes > task.max_kv_bytes || result_bytes == 0
        || result_bytes > task.max_output_bytes || result_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("sampled sentiment job complete KV and stored-result envelope").into());
    }
    let sampler = json_sampling_workspace_bytes().map_err(SampledSentimentError::from).map_err(task_error)?;
    let workspace_bytes = super::super::prefill::with_extra_workspace(sum(&[
        native.rope_bytes, native.scratch_payload_bound, limits.native.allocator_reserve_bytes, sampler,
    ])?, prefill)?;
    sum(&[input_bytes, native.kv_bytes, workspace_bytes])?;
    Ok(Requirements { native, workspace_bytes, input_bytes })
}

#[cfg(test)]
mod tests;
