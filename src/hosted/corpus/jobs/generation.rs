//! One resident native model and process-owned invocation for retained text.
use super::*;
use crate::{
    jobs::runner::generation::{Int8GenerationJobPlanner, Int8GenerationJobProcessor},
    native_engine::{simd::Int8Kernel, strict_int8::{prefill::Int8PrefillLimits,
        projection::Int8ProjectionExecutor}},
    tasks::ir::TaskBudget,
};

/// The dispatcher retains this complete owner outside its scoped callback.
/// On early return or unwind, readers, private preparation and every native
/// reservation remain charged until the last persistent worker has joined.
struct GenerationExecution<R> {
    input: Charged<StreamInput<(Int8GenerationJobPlanner, SourceJobRequest), R, ()>>,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    limits: JobHostLimits,
    required: Int8MemoryRequirement,
    prefill: Option<Int8PrefillLimits>,
    lease: EngineLease,
}

impl NlpEngine {
    /// Start or explicitly resume generation/chat over the COMPLETE original
    /// NDJSON population. The sealed planner owns the exact sampling policy;
    /// no caller-supplied admission provider or replacement native driver exists.
    /// Only completed validated outputs reach the authenticated result spool.
    /// Retention is explicit through SourceJobRequest, not a new default.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_text<R>(&self, model: &ResidentInt8, planner: Int8GenerationJobPlanner,
        request: SourceJobRequest, limits: JobHostLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_text_scheduled(model, planner, request, limits, None, None, reader, cancellation)
    }

    /// Retained generation/chat with explicitly bounded layer-major prompt
    /// execution. Extra scratch is charged before population IO and remains
    /// owned until the actual blocking closure drains. The schedule may change
    /// on resume; the authenticated semantic recipe and lifetime work cannot.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_text_layer_major<R>(&self, model: &ResidentInt8,
        planner: Int8GenerationJobPlanner, request: SourceJobRequest, limits: JobHostLimits,
        prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_text_scheduled(model, planner, request, limits, Some(prefill), None, reader, cancellation)
    }

    /// Keep one admitted numerical team across every pending generation/chat
    /// item in an authenticated retained job. Width includes the coordinator
    /// and must be in 2..=64. Complete output rows use the existing exact leaf
    /// kernels, and native work, addressed samples and attempts never renew.
    ///
    /// Team width is physical scheduling, so an explicit resume may change it
    /// without altering the frozen recipe or committed records. The same
    /// spool-first/journal-second runner owns all durable acknowledgements.
    /// Cancellation may leave committed progress; every child joins before
    /// any progress or error returns to the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_text_with_threads<R>(&self, model: &ResidentInt8,
        planner: Int8GenerationJobPlanner, request: SourceJobRequest, limits: JobHostLimits,
        kernel: Int8Kernel, compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_text_scheduled(model, planner, request, limits, None,
            Some((kernel, compute_threads)), reader, cancellation)
    }

    /// Compose bounded layer-major prompt morsels with one persistent INT8
    /// row team for the entire retained invocation. Fixed worker batch buffers
    /// are additional to the coordinator's prompt, KV and sampler storage.
    /// Neither prompt width nor team width changes the authenticated semantic
    /// recipe, original item ordinals or any lifetime debit on resume.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_text_layer_major_with_threads<R>(&self, model: &ResidentInt8,
        planner: Int8GenerationJobPlanner, request: SourceJobRequest, limits: JobHostLimits,
        kernel: Int8Kernel, compute_threads: usize, prefill: Int8PrefillLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_text_scheduled(model, planner, request, limits, Some(prefill),
            Some((kernel, compute_threads)), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn job_int8_text_scheduled<R>(&self, model: &ResidentInt8, planner: Int8GenerationJobPlanner,
        request: SourceJobRequest, limits: JobHostLimits, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match team {
            None => dispatch::preflight(self, limits.native.run)?,
            Some((_, width)) => dispatch::preflight_int8_team(self, limits.native.run, width)?,
        }
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), planner.execution_identity())?;
        let native = planner.native_limits();
        validate_work(native.max_model_work)?;
        let bytes = limits.required_buffer_bytes_for_input(request.limits, request.mode.input_memory())?;
        let required = requirements(limits.native)?;
        capacity(planner.task_budget(), required.kv_bytes, request.limits.max_result_bytes)?;
        let scratch_bytes = workspace(sum(&[required.rope_bytes, required.scratch_payload_bound,
            limits.native.allocator_reserve_bytes])?, native.max_sampler_bytes, prefill)?;
        let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
        let worker_bytes = team.map(|(_, width)|
            dispatch::int8_prefill_team_workspace_bytes(width, max_batch_rows)).transpose()?.unwrap_or(0);
        // The dispatcher reserves worker storage separately. Validate the
        // simultaneous envelope before any population or private job IO.
        sum(&[bytes, required.kv_bytes, scratch_bytes, worker_bytes])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let execution = GenerationExecution { input, kv: Some(kv), scratch: Some(scratch),
            _kv: None, _scratch: None, limits, required, prefill, lease };
        let model = model.clone();
        let completed = match team {
            None => dispatch::run(self, limits.native.run, cancellation, move |control| {
                let mut execution = execution;
                execution.run(&model.inner.loaded.value, None, None, control)
            }),
            Some((kernel, width)) => dispatch::run_int8_prefill_team(self, limits.native.run,
                cancellation, model, width, kernel, max_batch_rows, execution,
                move |model, execution, control, projection|
                    execution.run(model, Some(kernel), Some(projection), control)),
        }?;
        completed.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> GenerationExecution<R> {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::Limits("text job KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::Limits("text job scratch admission consumed"))?.commit()?);
        Ok(())
    }

    fn run<'weights>(&mut self, model: &'weights CurrentCandidateInt8Model,
        kernel: Option<Int8Kernel>, projection: Option<&'weights mut (dyn Int8ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        // Preserve the serial route's original checkpoint allowance. The
        // persistent route also checks its shared stop at phase boundaries.
        if projection.is_some() {
            if let Some(cause) = control.checkpoint(0) {
                return Err(HostedError::Native(StrictInt8Error::Cancelled(cause)));
            }
        }
        let (planner, request) = self.input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
        let population = match read_population(&mut self.input.value.reader, &request,
            self.limits.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(Err(JobRunError::Storage(error))),
        };
        if projection.is_some() {
            if let Some(cause) = control.checkpoint(0) {
                return Err(HostedError::Native(StrictInt8Error::Cancelled(cause)));
            }
        }
        let engine = model.engine(self.limits.native.context_tokens,
            memory_budget(self.required)).map_err(HostedError::Model)?;
        let engine = match kernel { Some(kernel) => engine.with_kernel(kernel), None => engine };
        let mut engine = match projection {
            Some(projection) => engine.with_projection_executor(projection).map_err(HostedError::Native)?,
            None => engine,
        };
        self.commit_native()?;
        let admission = CorpusAdmission { lease: &self.lease, model: model.artifact_identity(),
            kv_bytes: self.required.kv_bytes, sampler_bytes: planner.native_limits().max_sampler_bytes,
            output_bytes: request.limits.max_result_bytes as u64 };
        let result = (|| {
            let processor = match self.prefill {
                None => Int8GenerationJobProcessor::new(&planner, &mut engine, admission),
                Some(prefill) => Int8GenerationJobProcessor::new_layer_major(&planner,
                    &mut engine, admission, prefill),
            }.map_err(JobRunError::Processor)?;
            // Authenticate the same recipe and population before repair or
            // forwarding. Failed-attempt debits survive either schedule.
            run_population(request, &population, processor, control)
        })();
        // Native buffers and all runner borrows end here, but this admission
        // owner stays outside the team's scope until physical join completes.
        drop(engine); drop(population); drop(planner);
        Ok(result)
    }
}
fn capacity(task: TaskBudget, kv: u64, output: usize) -> Result<(), HostedError> {
    task.validate().map_err(|_| HostedError::Limits("text job task ceiling"))?;
    if kv == 0 || kv > task.max_kv_bytes || task.max_output_bytes > output as u64 {
        return Err(HostedError::Limits("text job complete KV or immutable output envelope"));
    }
    Ok(())
}
fn workspace(ordinary: u64, sampler: u64, prefill: Option<Int8PrefillLimits>) -> Result<u64, HostedError> {
    if sampler == 0 { return Err(HostedError::Limits("text job sampler reservation")); }
    let extra = prefill.map(Int8PrefillLimits::validate).transpose().map_err(HostedError::Native)?.unwrap_or(0);
    sum(&[ordinary, sampler, extra])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn task() -> TaskBudget {
        TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 65536,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
    }
    #[test]
    fn complete_kv_and_serialized_output_fit_before_population_ingestion() {
        capacity(task(), 1 << 30, 65536).unwrap();
        assert!(capacity(task(), (1 << 30) + 1, 65536).is_err());
        assert!(capacity(task(), 0, 65536).is_err());
        assert!(capacity(task(), 1 << 30, 65535).is_err());
    }
    #[test]
    fn sampler_workspace_is_additional_not_a_discount_on_native_scratch() {
        assert_eq!(workspace(12345, 32 << 20, None).unwrap(), 12345 + (32 << 20));
        assert!(workspace(12345, 0, None).is_err());
        assert!(workspace(u64::MAX, 1, None).is_err());
        assert_eq!(workspace(u64::MAX - 1, 1, None).unwrap(), u64::MAX);
    }
    #[test]
    fn grouped_prompt_workspace_is_validated_and_additional_for_the_whole_job() {
        for rows in [1, 4, 64] {
            let extra = Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let limits = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
            assert_eq!(workspace(12345, 32 << 20, Some(limits)).unwrap(), 12345 + (32 << 20) + extra);
            assert!(workspace(12345, 32 << 20, Some(Int8PrefillLimits {
                max_extra_scratch_bytes: extra - 1, ..limits })).is_err());
            assert!(workspace(u64::MAX - extra, 1, Some(limits)).is_err());
        }
        for rows in [0, 65, usize::MAX] {
            assert!(workspace(12345, 32 << 20, Some(Int8PrefillLimits {
                max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX })).is_err());
        }
    }
    #[test]
    fn invalid_task_budgets_are_not_repaired_to_match_storage() {
        let mut budget = task(); budget.max_output_tokens = 0;
        assert!(capacity(budget, 1 << 30, 65536).is_err());
        let mut budget = task(); budget.max_output_bytes = 0;
        assert!(capacity(budget, 1 << 30, 65536).is_err());
    }
    #[test]
    fn owned_planner_input_and_result_cross_the_existing_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<StreamInput<(Int8GenerationJobPlanner, SourceJobRequest), std::io::Cursor<Vec<u8>>, ()>>();
        send::<GenerationExecution<std::io::Cursor<Vec<u8>>>>();
        send::<crate::tasks::chat::quantized::Int8ChatResult>();
        let _entry = NlpEngine::job_int8_text::<std::io::Cursor<Vec<u8>>>;
        let _grouped = NlpEngine::job_int8_text_layer_major::<std::io::Cursor<Vec<u8>>>;
        let _team = NlpEngine::job_int8_text_with_threads::<std::io::Cursor<Vec<u8>>>;
        let _grouped_team = NlpEngine::job_int8_text_layer_major_with_threads::<std::io::Cursor<Vec<u8>>>;
    }
}
