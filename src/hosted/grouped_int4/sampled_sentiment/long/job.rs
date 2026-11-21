//! Whole-document independent sampling with authenticated retained input and
//! spool-first result publication on one resident, physically owned engine.

use super::*;
use std::{io::BufRead, panic::{catch_unwind, AssertUnwindSafe}, sync::Mutex};
use crate::{
    batch::sentiment::sampled::{SampledSentimentBatchLimits, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES},
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress, JobWork,
        runner::{JobRunError, sampled_sentiment::long::{SampledSentimentDocumentJobPlanner,
            SampledSentimentDocumentJobProcessor}}},
    tasks::sentiment::sampled::long::memory::MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
};
use super::corpus::DocumentAdmission;

/// Simultaneous admitted payload, excluding resident weights and the caller's
/// independently retained compiler. This is a checked quote, not measured RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampledSentimentDocumentJobRequirements {
    /// Whole-document preparation, native state, sampler and guarded result.
    /// Input additionally includes population/index/IO/journal staging.
    pub document: SampledSentimentMapMemoryRequirement,
    /// Separately reserved by the persistent projection-team dispatcher.
    pub team_workspace_bytes: u64,
}
impl SampledSentimentDocumentJobRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        let d = self.document;
        sum(&[d.input_bytes, d.native.kv_bytes, d.workspace_bytes,
            d.output_reservation_bytes, self.team_workspace_bytes])
    }
}

/// Pure complete-document durable admission before private source/key/model IO.
/// Per-chunk TaskBudget output does not cap the complete document: mapping and
/// JobLimits declare that separate envelope explicitly.
#[allow(clippy::too_many_arguments)]
pub fn int4_sampled_sentiment_document_job_requirements(width: usize, task: TaskBudget,
    planning: SampledSentimentLimits, mapping: SampledSentimentMapLimits,
    options: &SampledSentimentOptions, encoding: &UntrustedEncoding,
    work: SampledSentimentBatchLimits, job: JobLimits, host: JobHostLimits,
    input: JobInputMemory, prefill: Option<Int4PrefillLimits>, compute_threads: Option<usize>)
    -> Result<SampledSentimentDocumentJobRequirements, HostedJobError> {
    check_schedule(compute_threads, prefill)?;
    work.validate().map_err(HostedError::BatchSetup)?;
    let buffers = host.required_buffer_bytes_for_input(job, input)?;
    if job.max_work != JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run)
        || job.max_id_bytes > 256
        || mapping.max_model_work != work.max_item_work
        || mapping.chunks.max_input_bytes > job.max_input_bytes_per_item
        || mapping.max_result_bytes > job.max_result_bytes as u64
        || planning.max_input_bytes > job.max_input_bytes_per_item
        || task.max_output_bytes.min(planning.max_result_bytes) > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("sampled document job exact lifetime and document ceilings").into());
    }
    let mut document = SampledSentimentMapMemoryLimits {
        context_tokens: host.native.context_tokens,
        preparation_reserve_bytes: MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: host.native.allocator_reserve_bytes,
        max_axes: SentimentAxis::ALL.len(), max_sampler_bytes: work.max_sampler_bytes,
    }.requirements(width, task, planning, mapping, options, encoding).map_err(map_error)?;
    let preparation = document.input_bytes.checked_add(SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64)
        .ok_or(HostedError::Limits("sampled document job recipe arithmetic"))?;
    if preparation > host.preparation_reserve_bytes {
        return Err(HostedError::Limits("sampled document job complete preparation reservation").into());
    }
    document.input_bytes = buffers;
    document.workspace_bytes = super::super::super::prefill::with_extra_workspace(document.workspace_bytes, prefill)?;
    let team_workspace_bytes = compute_threads.map(|threads|
        int4_task_team_workspace_bytes(threads, width)).transpose()?.unwrap_or(0);
    let result = SampledSentimentDocumentJobRequirements { document, team_workspace_bytes };
    result.total_bytes()?;
    Ok(result)
}

struct JobInput<R> {
    preparation: Option<(SampledSentimentDocumentJobPlanner, SourceJobRequest)>,
    reader: R,
}
struct JobExecution<R> {
    input: Charged<JobInput<R>>, kv: Pending, scratch: Pending,
    limits: JobHostLimits, required: SampledSentimentDocumentJobRequirements, lease: EngineLease,
}
struct NativeClaims {
    kv: Option<Pending>, scratch: Option<Pending>,
    committed_kv: Option<CommittedMemory>, committed_scratch: Option<CommittedMemory>,
}
impl NativeClaims {
    fn commit(&mut self) -> Result<(), HostedError> {
        self.committed_kv = Some(self.kv.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        self.committed_scratch = Some(self.scratch.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        Ok(())
    }
}
struct DeferredDocumentJob<R> {
    result: Result<Result<JobProgress, JobRunError>, HostedError>,
    input: Charged<JobInput<R>>, native: NativeClaims, lease: EngineLease,
}
impl<R> DeferredDocumentJob<R> {
    fn finish(self, control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        checkpoint(control)?;
        let Self { result, input, native, lease } = self;
        drop(input); drop(native); drop(lease);
        result
    }
}
fn take_execution<R>(queued: &Mutex<Option<JobExecution<R>>>) -> Result<JobExecution<R>, HostedError> {
    queued.lock().map_err(|_| HostedError::Panicked)?.take().ok_or(HostedError::CompletionMissing)
}

impl NlpEngine {
    /// Start or resume complete-source independent sentiment sampling. Only
    /// complete validated maps commit; exact retained input replay is explicit
    /// in SourceJobRequest. Original document, chunk and sample identities
    /// survive resume while failed attempt work remains in the durable ledger.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sampled_sentiment_documents<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_sampled_sentiment_document_job(model, planner, request, limits,
            kernel, None, reader, cancellation)
    }

    /// One bounded causal prompt workspace; the selected physical schedule
    /// becomes part of the private authenticated recipe before storage opens.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sampled_sentiment_documents_layer_major<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, prefill: Int4PrefillLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let planner = planner.with_layer_major_prefill(prefill).map_err(HostedError::BatchSetup)?;
        self.run_int4_sampled_sentiment_document_job(model, planner, request, limits,
            kernel, None, reader, cancellation)
    }

    /// One fixed numerical team spans the complete population. Input, native
    /// claims and journal owners remain alive through actual worker join.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_sampled_sentiment_documents_with_threads<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_sampled_sentiment_document_job(model, planner, request, limits,
            kernel, Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_sampled_sentiment_document_job<R>(&self, model: &ResidentInt4,
        planner: SampledSentimentDocumentJobPlanner, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: Option<usize>,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let prefill = planner.prefill_limits();
        check_schedule(compute_threads, prefill)?;
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain.into()); }
        crate::native_engine::constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let required = int4_sampled_sentiment_document_job_requirements(planner.group_width(),
            planner.task_budget(), planner.planning_limits(), planner.mapping_limits(), planner.sampling_options(),
            planner.document_encoding(), planner.native_limits(), request.limits, limits,
            request.mode.input_memory(), prefill, compute_threads)?;
        if planner.preparation_requirement_bytes() > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("sampled document job retained compiler").into());
        }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.document.input_bytes)?,
            || Ok(JobInput { preparation: Some((planner, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.document.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.document.workspace_bytes)?;
        let queued_owner = Arc::new(Mutex::new(Some(JobExecution { input, kv, scratch, limits, required, lease })));
        let queued = Arc::clone(&queued_owner);
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control| {
                take_execution(&queued)?.run(&model.inner.loaded.value, kernel, None, control).finish(control)
            }),
            Some(width) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team| {
                    Ok(take_execution(&queued)?.run(weights, kernel, Some(team), control))
                }, |deferred, control| deferred.finish(control)),
        };
        drop(queued_owner);
        completed?.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> JobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> DeferredDocumentJob<R> {
        let Self { mut input, kv, scratch, limits, required, lease } = self;
        let mut native = NativeClaims { kv: Some(kv), scratch: Some(scratch),
            committed_kv: None, committed_scratch: None };
        let result = catch_unwind(AssertUnwindSafe(|| {
            checkpoint(control)?;
            let (planner, request) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let mapping = planner.mapping_limits(); let work = planner.native_limits();
            let population = match read_population(&mut input.value.reader, &request, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            let need = required.document;
            let engine = StrictInt4Engine::new(weights, limits.native.context_tokens, Int4MemoryBudget {
                max_kv_bytes: need.native.kv_bytes, max_rope_bytes: need.native.rope_bytes,
                max_scratch_payload_bytes: need.native.scratch_payload_bound }).map_err(native_error)?.with_kernel(kernel);
            let mut engine = match projection {
                Some(team) => engine.with_projection_executor(team).map_err(native_error)?,
                None => engine,
            };
            native.commit()?;
            let admission = DocumentAdmission::for_job(&lease, weights.identity(), limits.native.context_tokens,
                need.native.kv_bytes, mapping, work, limits.preparation_reserve_bytes,
                need.output_reservation_bytes, need.max_output_bytes, need.max_output_tokens);
            let result = (|| {
                let processor = SampledSentimentDocumentJobProcessor::new(planner, &mut engine, admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine); drop(population);
            Ok(result)
        })).unwrap_or(Err(HostedError::Panicked));
        DeferredDocumentJob { result, input, native, lease }
    }
}

fn check_schedule(threads: Option<usize>, prefill: Option<Int4PrefillLimits>) -> Result<(), HostedError> {
    if threads.is_some() && prefill.is_some() {
        return Err(HostedError::Limits("sampled document jobs use separate prefill and compute-team schedules"));
    }
    if let Some(limits) = prefill { limits.validate().map_err(native_error)?; }
    Ok(())
}

#[cfg(test)] mod tests;
