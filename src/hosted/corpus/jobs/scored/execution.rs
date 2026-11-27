//! One physical lifetime for authenticated INT8 jobs.
//! Private task adapters reuse the existing spool-first/journal-second runner.
use super::*;
use crate::native_engine::strict_int8::projection::Int8ProjectionExecutor;

pub(in crate::hosted::corpus) trait Int8JobTask: Send + Sized + 'static {
    fn identity(&self) -> &ExecutionIdentity;
    /// Actual immutable task assets survive runner errors and native unwind
    /// until the dispatcher has joined every physical child.
    fn retain_for_join(&self) -> Option<Box<dyn Send>> { None }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError>;
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> { Ok(0) }
    /// Exact live sampler payload, independently of its allowed policy cap.
    /// The dispatcher retains this scratch claim until all children join.
    fn sampler_workspace_bytes(&self) -> Result<u64, HostedError> { Ok(0) }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::hosted::corpus::jobs) struct Requirements {
    pub(in crate::hosted::corpus::jobs) native: Int8MemoryRequirement,
    pub(in crate::hosted::corpus::jobs) input_bytes: u64,
    pub(in crate::hosted::corpus::jobs) scratch_bytes: u64,
    pub(in crate::hosted::corpus::jobs) worker_bytes: u64,
    sampler_bytes: u64,
    max_batch_rows: usize,
}
impl Requirements {
    fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.input_bytes, self.native.kv_bytes, self.scratch_bytes, self.worker_bytes])
    }
}
#[allow(clippy::too_many_arguments)]
pub(in crate::hosted::corpus::jobs) fn quote(limits: JobHostLimits, job: JobLimits, input: JobInputMemory,
    native: Int8MemoryRequirement, task_bytes: u64, sampler_bytes: u64,
    prefill: Option<Int8PrefillLimits>, width: Option<usize>)
    -> Result<Requirements, HostedJobError> {
    let input_bytes = sum(&[limits.required_buffer_bytes_for_input(job, input)?, task_bytes])?;
    let scratch_bytes = crate::hosted::scored::scoring_scratch(sum(&[native.rope_bytes,
        native.scratch_payload_bound, limits.native.allocator_reserve_bytes, sampler_bytes])?, prefill)?;
    let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    let worker_bytes = width.map(|width|
        dispatch::int8_prefill_team_workspace_bytes(width, max_batch_rows)).transpose()?.unwrap_or(0);
    let required = Requirements { native, input_bytes, scratch_bytes, worker_bytes, sampler_bytes, max_batch_rows };
    required.total_bytes()?;
    Ok(required)
}

/// The dispatcher owns this aggregate OUTSIDE its scoped callback. Cancellation,
/// failed durable work, or unwind cannot release a reader, reservation or lease
/// while a child remains live. Progress becomes observable only after join.
struct RetainedInput<P, R> {
    _retained: Option<Box<dyn Send>>,
    population: Option<JobPopulation>,
    stream: StreamInput<(P, SourceJobRequest), R, ()>,
}
struct Execution<P, R> {
    input: Charged<RetainedInput<P, R>>,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    limits: JobHostLimits,
    required: Requirements,
    prefill: Option<Int8PrefillLimits>,
    lease: EngineLease,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::hosted::corpus) fn run<P, R>(host: &NlpEngine, model: &ResidentInt8,
    task: P, request: SourceJobRequest, limits: JobHostLimits,
    prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
    reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
where P: Int8JobTask, R: BufRead + Send + 'static {
    match team {
        None => dispatch::preflight(host, limits.native.run)?,
        Some((_, width)) => dispatch::preflight_int8_team(host, limits.native.run, width)?,
    }
    host.check_resident_domain(model)?;
    check_model_identity(model.artifact_identity(), task.identity())?;
    let native = requirements(limits.native)?;
    task.validate(native, request.limits)?;
    let required = quote(limits, request.limits, request.mode.input_memory(),
        native, task.preparation_workspace_bytes()?, task.sampler_workspace_bytes()?,
        prefill, team.map(|(_, width)| width))?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
        || {
            let retained = task.retain_for_join();
            Ok(RetainedInput { _retained: retained, population: None,
                stream: StreamInput { planner: Some((task, request)), reader, writer: () } })
        })?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
    let execution = Execution { input, kv: Some(kv), scratch: Some(scratch), _kv: None, _scratch: None,
        limits, required, prefill, lease };
    let model = model.clone();
    let completed = match team {
        None => dispatch::run(host, limits.native.run, cancellation, move |control| {
            let mut execution = execution;
            execution.run(&model.inner.loaded.value, None, None, control)
        }),
        Some((kernel, width)) => dispatch::run_int8_prefill_team(host, limits.native.run,
            cancellation, model, width, kernel, required.max_batch_rows, execution,
            move |model, execution, control, projection|
                execution.run(model, Some(kernel), Some(projection), control)),
    }?;
    completed.map_err(HostedJobError::Job)
}

impl<P: Int8JobTask, R: BufRead> Execution<P, R> {
    fn run<'weights>(&mut self, model: &'weights CurrentCandidateInt8Model,
        kernel: Option<Int8Kernel>, projection: Option<&'weights mut (dyn Int8ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let threaded = projection.is_some();
        if threaded {
            if let Some(cause) = control.checkpoint(0) {
                return Err(HostedError::Native(StrictInt8Error::Cancelled(cause)));
            }
        }
        let (task, request) = self.input.value.stream.planner.take().ok_or(HostedError::CompletionMissing)?;
        let population = match read_population(&mut self.input.value.stream.reader,
            &request, self.limits.transport, control) {
            Ok(Population::Original(original)) => {
                // The runner borrows these same authenticated original bytes;
                // its local teardown never releases them ahead of worker join.
                Population::BorrowedOriginal(self.input.value.population.insert(original))
            }
            Ok(population) => population,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        if threaded {
            if let Some(cause) = control.checkpoint(0) {
                return Err(HostedError::Native(StrictInt8Error::Cancelled(cause)));
            }
        }
        let engine = model.engine(self.limits.native.context_tokens,
            memory_budget(self.required.native)).map_err(HostedError::Model)?;
        let engine = match kernel { Some(kernel) => engine.with_kernel(kernel), None => engine };
        let mut engine = match projection {
            Some(projection) => engine.with_projection_executor(projection).map_err(HostedError::Native)?,
            None => engine,
        };
        // Population may retain the reader for protected input spooling.
        // Commit only disjoint claim fields, never reborrow the entire owner.
        self._kv = Some(self.kv.take()
            .ok_or(HostedError::Limits("job KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take()
            .ok_or(HostedError::Limits("job scratch admission consumed"))?.commit()?);
        let admission = CorpusAdmission { lease: &self.lease, model: model.artifact_identity(),
            kv_bytes: self.required.native.kv_bytes, sampler_bytes: self.required.sampler_bytes,
            output_bytes: request.limits.max_result_bytes as u64 };
        let result = task.execute(&mut engine, admission, request, &population, self.prefill, control);
        // Runner, journal and processor borrows have drained; claims remain in
        // the dispatcher-owned aggregate until physical worker join completes.
        drop(engine); drop(population);
        result
    }
}

#[cfg(test)]
mod tests;
