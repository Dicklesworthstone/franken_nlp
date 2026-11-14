//! Authenticated resumable NER, keyphrases, cited summaries and passage QA
//! on the actual resident grouped-INT4 engine.
use super::*;
use crate::{
    batch::source::grouped_int4::{Int4SourceAdmission, Int4SourceBatchAdmission},
    jobs::runner::source::grouped_int4::{Int4SourceJobPlanner, Int4SourceJobProcessor},
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
};

struct SourceJobInput<R> {
    preparation: Option<(Int4SourceJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}

// Capture the whole owner across the blocking boundary. Queued cancellation
// drops the private reader/factory/key before their charge and the lease last.
struct SourceJobExecution<R> {
    input: Charged<SourceJobInput<R>>,
    kv: Pending,
    scratch: Pending,
    limits: JobHostLimits,
    required: Requirements,
    lease: EngineLease,
}

/// Retain the caller-owned reader and native reservation/charges through
/// physical join on completed jobs and ordinary population/runner failures.
struct SourceJobCompletion<R> {
    result: Result<JobProgress, JobRunError>,
    input: Charged<SourceJobInput<R>>,
    native: NativeJobCharges,
    lease: EngineLease,
}

impl NlpEngine {
    /// Create or explicitly resume an owned source-task job from the complete
    /// original NDJSON population. The immutable factory fixes the task,
    /// actual model/group identity, compiler, defaults and lifetime work.
    ///
    /// The job owns its authenticated result spool. Input remains supplied by
    /// the caller on every invocation. Population ingestion, authentication,
    /// native execution, durable acknowledgement and optional materialization
    /// run inside one physical blocking invocation. No result text is emitted
    /// to stdout by this API.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source<R>(&self, model: &ResidentInt4, planner: Int4SourceJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_job(model, planner, vocabulary, request, limits,
            kernel, None, reader, cancellation)
    }

    /// Run the same authenticated source job with one persistent team of
    /// 2..=64 compute threads, including the coordinator. Every pending item,
    /// constrained token, durable commit and optional materialization shares
    /// the original invocation budget and one fixed numerical team.
    ///
    /// Worker width changes physical scheduling only. An explicit resume may
    /// choose another width without changing the private recipe or refunding
    /// persisted native/mask work and attempts. Worker scratch is separately
    /// admitted before population IO. Return waits for every worker to join;
    /// cancellation may still leave authenticated durable progress.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4SourceJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_job(model, planner, vocabulary, request, limits,
            kernel, Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_source_job<R>(&self, model: &ResidentInt4, planner: Int4SourceJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, compute_threads: Option<usize>,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
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
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        // The factory prices its complete planning context into task_ceiling.
        // Equality binds that context to the actual resident KV before any
        // population bytes are read or a private job directory is opened.
        let required = with_encoding_workspace(
            job_requirements(model.model_identity().group_width(), planner.task_ceiling(),
                request.limits, limits, request.mode.input_memory())?,
            planner.encoding_workspace_bytes())?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, model.model_identity().group_width()))
            .transpose()?.unwrap_or(0);
        // The dispatcher reserves this exact worker quote and retains it
        // through join. Check total arithmetic before reserving native inputs.
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes, team_scratch])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(SourceJobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let execution = SourceJobExecution { input, kv, scratch, limits, required, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_team(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        // Physical cancellation/drain failure takes precedence over a job
        // return value, including a durable commit with a lost acknowledgement.
        let SourceJobCompletion { result, input, native, lease } = completed;
        drop(input);
        drop(native);
        drop(lease);
        result.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> SourceJobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<SourceJobCompletion<R>, HostedError> {
        let Self { mut input, kv, scratch, limits, required, lease } = self;
        checkpoint(control)?;
        let (planner, vocabulary, request) = input.value.preparation.take()
            .ok_or(HostedError::CompletionMissing)?;
        let native_work = planner.native_limits();
        let population = match read_population(&mut input.value.reader, &request,
            limits.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(SourceJobCompletion {
                result: Err(JobRunError::Storage(error)), input,
                native: NativeJobCharges::Pending { _kv: kv, _scratch: scratch }, lease,
            }),
        };
        checkpoint(control)?;
        let mut engine = allocate_native(kv, scratch, || {
            let engine = StrictInt4Engine::new(weights, limits.native.context_tokens, Int4MemoryBudget {
                max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound,
            }).map_err(native_error)?.with_kernel(kernel);
            match projection {
                Some(team) => engine.with_projection_executor(team).map_err(native_error),
                None => Ok(engine),
            }
        })?;
        let admission = SourceJobAdmission { inner: CorpusAdmission {
            lease: &lease, model: weights.identity(), kv_bytes: required.native.kv_bytes,
            masks: native_work.masks, output_bytes: request.limits.max_result_bytes as u64,
            selected_rows: None,
        }, task_spec: &planner.execution_identity().task_spec };
        let result = (|| {
            let processor = Int4SourceJobProcessor::new(&planner, &mut engine.value, &vocabulary, admission)
                .map_err(JobRunError::Processor)?;
            // Both schedules use the same sealed processor and runner.
            // Authentication precedes replay/tail repair; native output guards
            // stay intact through spool sync and journal acknowledgement.
            run_population(request, &population, processor, control)
        })();
        let ChargedNative { value, _kv, _scratch } = engine;
        drop(value);
        drop(population);
        drop(vocabulary);
        drop(planner);
        Ok(SourceJobCompletion {
            result, input, native: NativeJobCharges::Committed { _kv, _scratch }, lease,
        })
    }
}

// One transient encoder workspace coexists with the charged population and
// native state. The original planner Arc keeps its cumulative ledger; this
// host neither forks it per record nor refunds preparation/reconstruction.
fn with_encoding_workspace(mut required: Requirements, workspace: u64)
    -> Result<Requirements, HostedError> {
    required.input_bytes = sum(&[required.input_bytes, workspace])?;
    sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes])?;
    Ok(required)
}

struct SourceJobAdmission<'a> { inner: CorpusAdmission<'a>, task_spec: &'a str }
impl Int4SourceBatchAdmission for SourceJobAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4SourceAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        self.inner.admit_for_task(request, self.task_spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn private_population_factory_and_result_owners_cross_the_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<Int4SourceJobPlanner>();
        send::<SourceJobInput<Cursor<Vec<u8>>>>();
        send::<Charged<SourceJobInput<Cursor<Vec<u8>>>>>();
        send::<SourceJobExecution<Cursor<Vec<u8>>>>();
        send::<SourceJobCompletion<Cursor<Vec<u8>>>>();
        let _entry = NlpEngine::job_int4_source::<Cursor<Vec<u8>>>;
        let _team_entry = NlpEngine::job_int4_source_with_threads::<Cursor<Vec<u8>>>;
    }
}
