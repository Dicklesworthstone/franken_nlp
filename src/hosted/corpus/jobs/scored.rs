//! Durable scored corpora on the existing process-owned native invocation.
use super::*;
use crate::{
    batch::classify::quantized::Int8ClassificationAdmission,
    jobs::runner::scored::{Int8ClassificationJobPlanner, Int8ClassificationJobProcessor,
        Int8SentimentJobPlanner, Int8SentimentJobProcessor},
    tasks::{classify::ClassificationPlanner, sentiment::SentimentPlanner, ir::TaskBudget},
};

impl NlpEngine {
    /// Start/resume exact exclusive or independent-label native scoring. The
    /// original population and private recipe authenticate before any job-file
    /// repair or model forward. No caller-supplied admission provider exists.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_classify<R>(&self, model: &ResidentInt8, planner: Arc<ClassificationPlanner>,
        config: ClassificationCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        validate_work(config.max_model_work)?;
        let bytes = limits.reservation_bytes(request.limits)?;
        let required = requirements(limits.native)?;
        let output_bytes = capacity(config.task_ceiling, required.kv_bytes, request.limits.max_result_bytes)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, config, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Keep the WHOLE input owner, including queued cancellation paths.
            let mut input = input;
            let (planner, config, request) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let factory = Int8ClassificationJobPlanner::new(&planner, config.identity,
                config.task_ceiling, config.planning, config.defaults, config.max_model_work)
                .map_err(HostedError::BatchSetup)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(p) => p, Err(e) => return Ok(Err(JobRunError::Storage(e))),
            };
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = ClassificationJobAdmission { output_bytes, inner: CorpusAdmission {
                lease: &lease, model: model.artifact_identity(), kv_bytes: required.kv_bytes,
                sampler_bytes: 0, output_bytes: request.limits.max_result_bytes as u64 } };
            let result = (|| {
                let processor = Int8ClassificationJobProcessor::new(factory, &mut engine.value, admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            // The runner/journal/native borrow drops before engine and inputs;
            // each output guard already survived spool sync and commit.
            drop(engine); drop(population); drop(planner); drop(input); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }

    /// One complete independent-axis result per durable item. Resume skips
    /// committed items; partial-axis failures never become successful records.
    /// Mode/EOS/policy remain bound to the actual immutable sentiment planner.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sentiment<R>(&self, model: &ResidentInt8, planner: Arc<SentimentPlanner>,
        config: SentimentCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        config.validate(&planner).map_err(HostedError::BatchSetup)?;
        let bytes = limits.reservation_bytes(request.limits)?;
        let required = requirements(limits.native)?;
        capacity(config.task_ceiling, required.kv_bytes, request.limits.max_result_bytes)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, config, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let (planner, config, request) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let factory = Int8SentimentJobPlanner::new(&planner, config).map_err(HostedError::BatchSetup)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(p) => p, Err(e) => return Ok(Err(JobRunError::Storage(e))),
            };
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0, output_bytes: request.limits.max_result_bytes as u64 };
            let result = (|| {
                let processor = Int8SentimentJobProcessor::new(factory, &mut engine.value, admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine); drop(population); drop(planner); drop(input); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}

// Concrete process admission, private to this host; never a caller-built permit.
struct ClassificationJobAdmission<'a> { inner: CorpusAdmission<'a>, output_bytes: u64 }
impl Int8ClassificationAdmission for ClassificationJobAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, identity: &ExecutionIdentity, work: Int8Work)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        if identity.task_spec != "classify-v1" { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        self.inner.admit_output(identity, work, self.inner.kv_bytes, 0, self.output_bytes)
    }
}
fn capacity(task: TaskBudget, kv: u64, result: usize) -> Result<u64, HostedError> {
    task.validate().map_err(|_| HostedError::Limits("scored job task ceiling"))?;
    if kv == 0 || kv > task.max_kv_bytes || task.max_output_bytes > result as u64 {
        return Err(HostedError::Limits("scored job full KV and immutable result envelope"));
    }
    Ok(task.max_output_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn budget() -> TaskBudget {
        TaskBudget { max_input_tokens: 2048, max_output_tokens: 16,
            max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
    }
    #[test]
    fn complete_resident_kv_and_task_result_ceiling_are_admitted_before_ingestion() {
        assert_eq!(capacity(budget(), 1 << 30, 65536).unwrap(), 65536);
        assert!(capacity(budget(), (1 << 30) + 1, 65536).is_err());
        assert!(capacity(budget(), 0, 65536).is_err());
        assert!(capacity(budget(), 1 << 30, 65535).is_err());
        let mut b = budget(); b.max_output_bytes = 0;
        assert!(capacity(b, 1 << 30, 65536).is_err());
    }
    #[test]
    fn both_owned_inputs_cross_the_existing_send_runtime_boundary() {
        fn send<T: Send + 'static>() {}
        send::<StreamInput<(Arc<ClassificationPlanner>, ClassificationCorpusConfig, SourceJobRequest),
            std::io::Cursor<Vec<u8>>, ()>>();
        send::<StreamInput<(Arc<SentimentPlanner>, SentimentCorpusConfig, SourceJobRequest),
            std::io::Cursor<Vec<u8>>, ()>>();
    }
}
