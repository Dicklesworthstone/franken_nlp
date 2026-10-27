//! Owned resumable classification, sentiment and judgment on resident INT4.
//! The existing job runner owns authenticated replay, durable work debits and
//! spool-first publication; this host owns all physical execution resources.
use super::*;
use std::io::BufRead;
use crate::{
    batch::scored_int4::Int4ScoredBatchLimits,
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, run_population},
    jobs::{JobLimits, JobProgress, population::JobPopulation,
        runner::{JobRunError, scored::grouped_int4::{Int4ScoredJobPlanner,
            Int4ScoredJobProcessor, Int4ScoredJobTask}}},
    native_engine::constrained_int4,
    tasks::ir::TaskBudget,
};
use super::scored_batch::ScoredAdmission;

struct JobInput<P: Int4ScoredJobTask, R> {
    preparation: Option<(Int4ScoredJobPlanner<P>, SourceJobRequest)>,
    reader: R,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Requirements { native: Int4MemoryRequirement, workspace_bytes: u64, input_bytes: u64 }

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
    /// the existing owner-only job protocol. No input spool, key generation,
    /// automatic retry or result publication to stdout is introduced here.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_scored<P, R>(&self, model: &ResidentInt4,
        planner: Int4ScoredJobPlanner<P>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where P: Int4ScoredJobTask, R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
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
            work, request.limits, limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(JobInput { preparation: Some((planner, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the entire owner: queued cancellation must drop owned
            // inputs, private settings and key before their buffer charge.
            let mut input = input;
            checkpoint(control)?;
            let (planner, request) = input.value.preparation.take()
                .ok_or(HostedError::CompletionMissing)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader,
                &request.key, request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                        max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = ScoredAdmission::for_job(&lease, model.model_identity(),
                limits.native.context_tokens, required.native.kv_bytes,
                work.max_item_work, P::TASK, request.limits.max_result_bytes as u64);
            let result = (|| {
                let processor = Int4ScoredJobProcessor::new(planner, &mut engine.value, admission)
                    .map_err(JobRunError::Processor)?;
                // The runner validates the immutable recipe before storage
                // opens, syncs work before execution, and holds each output
                // guard through spool sync and journal acknowledgement.
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            drop(population);
            drop(input);
            drop(lease);
            Ok(result)
        })?;
        // Physical drain/cancellation failure takes precedence over a job
        // result. Durable progress is reconciled by authenticated resume.
        completed.map_err(HostedJobError::Job)
    }
}

fn requirements(width: usize, task: TaskBudget, work: Int4ScoredBatchLimits,
    job: JobLimits, limits: JobHostLimits) -> Result<Requirements, HostedJobError> {
    task.validate().map_err(|_| HostedError::Limits("INT4 scored job task ceiling"))?;
    work.validate().map_err(HostedError::BatchSetup)?;
    let input_bytes = limits.required_buffer_bytes(job)?;
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
