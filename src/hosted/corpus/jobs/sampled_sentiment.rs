//! Authenticated, resumable INT8 sampled-sentiment bundles on one resident model.
//! The existing durable runner owns exact inputs, private recipes and lifetime
//! debits; the shared dispatcher keeps every physical owner until worker join.
use super::*;
use crate::{
    jobs::{JobWork, runner::sampled_sentiment::quantized::{
        Int8SampledSentimentJobAdmission, Int8SampledSentimentJobLimits,
        Int8SampledSentimentJobPlanner, Int8SampledSentimentJobProcessor,
    }},
    native_engine::constrained_int8::json_sampling_workspace_bytes,
    tasks::ir::TaskBudget,
};

/// Complete simultaneous host commitment, excluding loaded model weights,
/// independently retained caller preparation and the runtime cleanup reserve.
/// This quote grants no memory and makes no measured-RSS assertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SampledSentimentJobRequirements {
    pub native: Int8MemoryRequirement,
    /// Population or retained-record staging, planner/vocabulary/recipe, IO,
    /// journal and serialization owners. Preparation is charged exactly once.
    pub input_bytes: u64,
    /// Native scratch, RoPE, allocator, exact sampler payload and prompt morsel.
    pub workspace_bytes: u64,
    /// Separate whole-bundle output claim through spool/journal acknowledgement.
    pub output_reservation_bytes: u64,
    /// One persistent row team, or zero for coordinator-only execution.
    pub team_workspace_bytes: u64,
}
impl Int8SampledSentimentJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.input_bytes, self.native.kv_bytes, self.workspace_bytes,
            self.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Price the complete job before opening private population or model files.
/// The preparation quote includes the compiler, vocabulary, largest plan and
/// frozen recipe and must fit the explicitly supplied preparation reserve.
/// The task token ceiling covers the entire axis-by-sample bundle. Worker
/// width is physical; the optional prompt schedule is frozen by the planner.
#[allow(clippy::too_many_arguments)]
pub fn int8_sampled_sentiment_job_requirements(task: TaskBudget,
    work: Int8SampledSentimentJobLimits, preparation_bytes: u64, job: JobLimits,
    host: JobHostLimits, input: JobInputMemory, prefill: Option<Int8PrefillLimits>,
    compute_threads: Option<usize>) -> Result<Int8SampledSentimentJobRequirements, HostedJobError> {
    task.validate().map_err(|_| HostedError::Limits("sampled INT8 job task ceiling"))?;
    work.validate().map_err(HostedError::BatchSetup)?;
    if job.max_work != (JobWork { model: work.max_model_work,
        mask_node_visits: work.masks.max_visits_per_run }) {
        return Err(HostedError::Limits("sampled INT8 job lifetime native and mask work").into());
    }
    if preparation_bytes == 0 || preparation_bytes > host.preparation_reserve_bytes {
        return Err(HostedError::Limits("sampled INT8 job complete preparation reserve").into());
    }
    let native = requirements(host.native)?;
    let result_bytes = task.max_output_bytes.min(job.max_result_bytes as u64);
    if native.kv_bytes > task.max_kv_bytes || result_bytes == 0 {
        return Err(HostedError::Limits("sampled INT8 job complete KV and stored-result envelope").into());
    }
    let sampler_bytes = json_sampling_workspace_bytes()
        .map_err(|error| HostedError::SampledSentiment(error.into()))?;
    if sampler_bytes > work.max_sampler_bytes {
        return Err(HostedError::Limits("sampled INT8 job sampler payload").into());
    }
    // Use the same geometry as execution; the complete preparation already
    // lives in JobHostLimits and must never be added a second time.
    let physical = scored::quote(host, job, input, native, 0, sampler_bytes, prefill, compute_threads)?;
    let output_reservation_bytes = result_bytes.checked_mul(4)
        .and_then(|bytes| bytes.checked_add(u64::from(task.max_output_tokens) * 8))
        .and_then(|bytes| bytes.checked_add(4096))
        .ok_or(HostedError::Limits("sampled INT8 job output arithmetic"))?;
    let required = Int8SampledSentimentJobRequirements { native: physical.native,
        input_bytes: physical.input_bytes, workspace_bytes: physical.scratch_bytes,
        output_reservation_bytes, team_workspace_bytes: physical.worker_bytes };
    required.total_bytes()?;
    Ok(required)
}

impl NlpEngine {
    /// Start or explicitly resume complete, independently addressed sentiment
    /// bundles. Original and retained input modes use the existing authenticated
    /// job protocol. Every attempt is charged before execution and only fully
    /// validated bundles can commit. The planner's private seed, task policy,
    /// tokenizer encoding and optional prompt schedule remain fixed on resume.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sampled_sentiment<R>(&self, model: &ResidentInt8,
        planner: Arc<Int8SampledSentimentJobPlanner>, request: SourceJobRequest,
        limits: JobHostLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_sampled_sentiment_scheduled(model, planner, request, limits,
            None, reader, cancellation)
    }

    /// Reuse one sealed projection team across every item, axis and sample,
    /// including the planner's optional layer-major prompt schedule. Immutable
    /// task assets, exact population and every native claim survive callback
    /// failure, cancellation and unwind until the last child physically joins.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sampled_sentiment_with_threads<R>(&self, model: &ResidentInt8,
        planner: Arc<Int8SampledSentimentJobPlanner>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: Int8Kernel, compute_threads: usize,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_sampled_sentiment_scheduled(model, planner, request, limits,
            Some((kernel, compute_threads)), reader, cancellation)
    }

    /// Select the physical worker arrangement without granting a caller-built
    /// native driver or changing the authenticated sampling recipe.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sampled_sentiment_scheduled<R>(&self, model: &ResidentInt8,
        planner: Arc<Int8SampledSentimentJobPlanner>, request: SourceJobRequest,
        limits: JobHostLimits, team: Option<(Int8Kernel, usize)>,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let prefill = planner.prefill_limits();
        int8_sampled_sentiment_job_requirements(planner.task_budget(), planner.native_limits(),
            planner.preparation_requirement_bytes(), request.limits, limits,
            request.mode.input_memory(), prefill, team.map(|(_, width)| width))?;
        scored::run(self, model, SampledTask { planner }, request, limits,
            prefill, team, reader, cancellation)
    }
}

struct SampledTask { planner: Arc<Int8SampledSentimentJobPlanner> }
impl scored::Int8JobTask for SampledTask {
    fn identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        Some(Box::new(Arc::clone(&self.planner)))
    }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        self.planner.check_job_limits(job).map_err(HostedError::BatchSetup)?;
        if native.kv_bytes > self.planner.task_budget().max_kv_bytes
            || self.planner.sampler_workspace_bytes() > self.planner.native_limits().max_sampler_bytes {
            return Err(HostedError::Limits("sampled INT8 job native admission"));
        }
        Ok(())
    }
    fn sampler_workspace_bytes(&self) -> Result<u64, HostedError> {
        Ok(self.planner.sampler_workspace_bytes())
    }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let schedule = |limits: Option<Int8PrefillLimits>|
            limits.map(|limits| (limits.max_batch_rows, limits.max_extra_scratch_bytes));
        if schedule(prefill) != schedule(self.planner.prefill_limits()) {
            return Err(HostedError::Limits("sampled INT8 job frozen prompt schedule"));
        }
        let processor = Int8SampledSentimentJobProcessor::new(self.planner, engine, admission)
            .map_err(HostedError::BatchSetup)?;
        Ok(run_population(request, population, processor, control))
    }
}

impl Int8SampledSentimentJobAdmission for CorpusAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, identity: &ExecutionIdentity, work: Int8Work,
        kv_bytes: u64, sampler_bytes: u64, max_result_bytes: u64)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        // This guard owns the complete bundle and token/JSON staging through
        // result-spool sync and the durable journal acknowledgement.
        self.admit_output(identity, work, kv_bytes, sampler_bytes, max_result_bytes)
    }
}

#[cfg(test)]
mod tests;
