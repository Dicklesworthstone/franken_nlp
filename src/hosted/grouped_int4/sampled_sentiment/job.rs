//! Owned resumable sampled-sentiment bundles on one resident native model.
//! The existing job protocol authenticates the complete population and frozen
//! private sampler recipe, debits work durably and publishes whole bundles.
//! This host retains native, input and output ownership through physical drain.

use super::*;
use std::{io::BufRead, panic::{catch_unwind, AssertUnwindSafe}, sync::Mutex};
use crate::{
    batch::sentiment::sampled::SampledSentimentBatchLimits,
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork,
        runner::{JobRunError, sampled_sentiment::{SampledSentimentJobPlanner, SampledSentimentJobProcessor}}},
    native_engine::{constrained_int4, strict_int4::projection::Int4ProjectionExecutor},
};
use super::corpus::Admission;

/// Modeled simultaneous payload, excluding resident model weights, the
/// caller's independently retained preparation and the runtime cleanup reserve.
/// This is a checked admission quote, not an RSS measurement or a memory grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampledSentimentJobRequirements {
    pub native: Int4MemoryRequirement,
    /// Population/index/parser, planner/vocabulary/recipe, IO, journal and
    /// serialization owners. Retained-input modes price one input record.
    pub input_bytes: u64,
    /// RoPE, native scratch, allocator, sampler and declared prompt morsel.
    pub workspace_bytes: u64,
    /// Maximum separate per-item result claim through durable acknowledgement.
    pub output_reservation_bytes: u64,
    /// One persistent projection team; zero for coordinator-only execution.
    pub team_workspace_bytes: u64,
}
impl SampledSentimentJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.input_bytes, self.native.kv_bytes, self.workspace_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Check the complete durable envelope before opening private job/input/model
/// files. The output claim covers four copies of the smaller task/job result
/// ceiling plus 4096 bytes; a planner may impose an even tighter result bound.
/// The preparation argument is the job planner's complete quote, already
/// including compact-tokenizer and frozen-recipe workspace. The returned total
/// includes a projection team exactly once.
#[allow(clippy::too_many_arguments)]
pub fn int4_sampled_sentiment_job_requirements(width: usize, task: TaskBudget,
    work: SampledSentimentBatchLimits, preparation_bytes: u64, job: JobLimits,
    host: JobHostLimits, input: JobInputMemory, prefill: Option<Int4PrefillLimits>,
    compute_threads: Option<usize>) -> Result<SampledSentimentJobRequirements, HostedJobError> {
    check_schedule(compute_threads, prefill)?;
    let mut required = requirements(width, task, work, preparation_bytes,
        task.max_output_bytes.min(job.max_result_bytes as u64), job, host, input, prefill)?;
    required.team_workspace_bytes = compute_threads.map(|threads|
        int4_task_team_workspace_bytes(threads, width)).transpose()?.unwrap_or(0);
    required.total_bytes()?;
    Ok(required)
}

struct JobInput<R> {
    preparation: Option<(SampledSentimentJobPlanner, SourceJobRequest)>,
    reader: R,
}

struct JobExecution<R> {
    input: Charged<JobInput<R>>, kv: Pending, scratch: Pending,
    limits: JobHostLimits, required: SampledSentimentJobRequirements, lease: EngineLease,
}

/// Claims stay outside every fallible native operation and its unwind boundary.
/// Native storage itself drops before this aggregate leaves the coordinator.
struct RetainedNativeClaims {
    kv: Option<Pending>,
    scratch: Option<Pending>,
    committed_kv: Option<CommittedMemory>,
    committed_scratch: Option<CommittedMemory>,
}
impl RetainedNativeClaims {
    fn commit(&mut self) -> Result<(), HostedError> {
        self.committed_kv = Some(self.kv.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        self.committed_scratch = Some(self.scratch.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        Ok(())
    }
}

/// Both error layers travel with their owners. Setup refusal, cancellation,
/// reader/native panic and successful progress all retain the same charges
/// until the projection team has physically joined.
struct DeferredSampledJob<R> {
    result: Result<Result<JobProgress, JobRunError>, HostedError>,
    input: Charged<JobInput<R>>,
    native: RetainedNativeClaims,
    lease: EngineLease,
}
impl<R> DeferredSampledJob<R> {
    fn finish(self, control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        checkpoint(control)?;
        let Self { result, input, native, lease } = self;
        drop(input); drop(native); drop(lease);
        result
    }
}

fn take_execution<R>(queued: &Mutex<Option<JobExecution<R>>>) -> Result<JobExecution<R>, HostedError> {
    // The lock ends before any input, allocation or numerical work begins.
    queued.lock().map_err(|_| HostedError::Panicked)?
        .take().ok_or(HostedError::CompletionMissing)
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
    /// `SampledSentimentJobPlanner::with_reused_prefill` explicitly freezes
    /// per-axis prompt reuse into the authenticated recipe. Reuse is local to
    /// each item and its returned bundle records actual work; the durable
    /// ledger still reserves the complete cold attempt before execution.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sampled_sentiment<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_sampled_sentiment_job(model, planner, request, limits, kernel,
            None, reader, cancellation)
    }

    /// Freeze one admitted causal prompt workspace into the private recipe.
    /// Its rows and declared capacity must match on authenticated resume.
    /// Each sample retains an independent address. If the planner explicitly
    /// enables reuse, only the current axis's first prompt pays prefill work;
    /// every subsequent sample resumes that exact prompt in the same session.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sampled_sentiment_layer_major<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, prefill: Int4PrefillLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let planner = planner.with_layer_major_prefill(prefill).map_err(HostedError::BatchSetup)?;
        self.run_int4_sampled_sentiment_job(model, planner, request, limits, kernel,
            None, reader, cancellation)
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
            Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_sampled_sentiment_job<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: Option<usize>,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let prefill = planner.prefill_limits();
        check_schedule(compute_threads, prefill)?;
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width()
            || planner.base_execution_identity().task_spec != "sentiment-v1"
            || planner.base_execution_identity().sampler_version != constrained_int4::JSON_SAMPLER_VERSION {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let result_cap = planner.task_budget().max_output_bytes.min(request.limits.max_result_bytes as u64);
        if planner.max_result_bytes() == 0 || planner.max_result_bytes() > result_cap {
            return Err(HostedError::Limits("sampled sentiment job effective result envelope").into());
        }
        let required = int4_sampled_sentiment_job_requirements(planner.group_width(),
            planner.task_budget(), planner.native_limits(), planner.preparation_requirement_bytes(),
            request.limits, limits, request.mode.input_memory(), prefill, compute_threads)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(JobInput { preparation: Some((planner, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let execution = JobExecution { input, kv, scratch, limits, required, lease };
        // Formation and pre-work cancellation may discard the callback before
        // it runs. This second owner survives the entire physical dispatch,
        // so that path also retains private input and native reservations.
        let queued_owner = Arc::new(Mutex::new(Some(execution)));
        let queued = Arc::clone(&queued_owner);
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control| {
                let execution = take_execution(&queued)?;
                execution.run(&model.inner.loaded.value, kernel, None, control).finish(control)
            }),
            Some(width) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team| {
                    let execution = take_execution(&queued)?;
                    Ok(execution.run(weights, kernel, Some(team), control))
                }, |deferred, control| deferred.finish(control)),
        };
        drop(queued_owner);
        completed?.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> JobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> DeferredSampledJob<R> {
        let Self { mut input, kv, scratch, limits, required, lease } = self;
        let mut native = RetainedNativeClaims { kv: Some(kv), scratch: Some(scratch),
            committed_kv: None, committed_scratch: None };
        // Borrow owners across this boundary. Any early error or unwind is
        // stored in the deferred value, never returned around its lifetime.
        let result = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let (planner, request) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let work = planner.native_limits();
            let result_bytes = planner.max_result_bytes();
            let population = match read_population(&mut input.value.reader, &request, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            // Construct before consuming either claim: an allocation refusal
            // must leave both native reservations owned until physical join.
            let engine = StrictInt4Engine::new(weights, limits.native.context_tokens,
                Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound })
                .map_err(native_error)?.with_kernel(kernel);
            let mut engine = match projection {
                Some(team) => engine.with_projection_executor(team).map_err(native_error)?,
                None => engine,
            };
            native.commit()?;
            let admission = Admission::for_job(&lease, weights.identity(), limits.native.context_tokens,
                required.native.kv_bytes, work, limits.preparation_reserve_bytes, result_bytes,
                planner.selected_rows());
            let result = (|| {
                let processor = SampledSentimentJobProcessor::new(planner, &mut engine, admission)
                    .map_err(JobRunError::Processor)?;
                // Whole bundle publication retains its output guard through
                // spool synchronization and the journal acknowledgement.
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            drop(population);
            Ok(result)
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredSampledJob { result, input, native, lease }
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
    input: JobInputMemory, prefill: Option<Int4PrefillLimits>)
    -> Result<SampledSentimentJobRequirements, HostedJobError> {
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
    let output_reservation_bytes = result_bytes.checked_mul(4).and_then(|bytes| bytes.checked_add(4096))
        .ok_or(HostedError::Limits("sampled sentiment job output arithmetic"))?;
    let required = SampledSentimentJobRequirements { native, workspace_bytes, input_bytes,
        output_reservation_bytes, team_workspace_bytes: 0 };
    required.total_bytes()?;
    Ok(required)
}

#[cfg(test)]
mod tests;
