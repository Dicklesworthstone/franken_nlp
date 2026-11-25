//! Durable scored corpora on the existing process-owned native invocation.
use super::*;
use crate::{
    batch::classify::quantized::Int8ClassificationAdmission,
    jobs::runner::scored::{EncodingWorkScope, Int8ClassificationJobPlanner, Int8ClassificationJobProcessor,
        Int8SentimentJobPlanner, Int8SentimentJobProcessor},
    tasks::{classify::ClassificationPlanner, sentiment::SentimentPlanner, ir::TaskBudget},
    tokenizer::untrusted::SafeBpeWorkLimits,
};

mod execution;
pub(in crate::hosted::corpus) use execution::{run, Int8JobTask};
use crate::native_engine::{simd::Int8Kernel, strict_int8::{StrictInt8Engine, prefill::Int8PrefillLimits}};

impl NlpEngine {
    /// Start/resume complete exclusive or independent-label scoring with the
    /// existing authenticated population, semantic recipe and durable debits.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_classify<R>(&self, model: &ResidentInt8, planner: Arc<ClassificationPlanner>,
        config: ClassificationCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_classify_scheduled(model, planner, config, request, limits,
            None, None, None, reader, cancellation)
    }

    /// Bound every classification head's prompt without changing the private
    /// semantic recipe. Prompt width may change on authenticated resume.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_classify_layer_major<R>(&self, model: &ResidentInt8, planner: Arc<ClassificationPlanner>,
        config: ClassificationCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_classify_scheduled(model, planner, config, request, limits,
            None, Some(prefill), None, reader, cancellation)
    }

    /// Reuse one admitted projection team across every pending record and
    /// scoring head. Width is physical scheduling, not new work authority.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_classify_with_threads<R>(&self, model: &ResidentInt8, planner: Arc<ClassificationPlanner>,
        config: ClassificationCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        kernel: Int8Kernel, compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_classify_scheduled(model, planner, config, request, limits,
            None, None, Some((kernel, compute_threads)), reader, cancellation)
    }

    /// Compose packed prompts with one persistent team. Coordinator and child
    /// storage coexist through physical drain; neither width renews attempts.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_classify_layer_major_with_threads<R>(&self, model: &ResidentInt8,
        planner: Arc<ClassificationPlanner>, config: ClassificationCorpusConfig,
        request: SourceJobRequest, limits: JobHostLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_classify_scheduled(model, planner, config, request, limits,
            None, Some(prefill), Some((kernel, compute_threads)), reader, cancellation)
    }

    /// Commit one complete independently scored axis bundle per durable item.
    /// Committed records are skipped and failed attempts remain debited.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sentiment<R>(&self, model: &ResidentInt8, planner: Arc<SentimentPlanner>,
        config: SentimentCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_sentiment_scheduled(model, planner, config, request, limits,
            None, None, None, reader, cancellation)
    }

    /// Bound prompts for every axis while retaining the unchanged semantic
    /// recipe, original ordinals and all durable native work counters.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sentiment_layer_major<R>(&self, model: &ResidentInt8, planner: Arc<SentimentPlanner>,
        config: SentimentCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_sentiment_scheduled(model, planner, config, request, limits,
            None, Some(prefill), None, reader, cancellation)
    }

    /// A fixed projection team spans every pending document and complete axis
    /// bundle. Every worker joins before progress or failure is returned.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sentiment_with_threads<R>(&self, model: &ResidentInt8, planner: Arc<SentimentPlanner>,
        config: SentimentCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        kernel: Int8Kernel, compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_sentiment_scheduled(model, planner, config, request, limits,
            None, None, Some((kernel, compute_threads)), reader, cancellation)
    }

    /// Share admitted worker batch capacity across packed prompts and candidate
    /// continuations. Explicit resume may change these physical widths.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sentiment_layer_major_with_threads<R>(&self, model: &ResidentInt8,
        planner: Arc<SentimentPlanner>, config: SentimentCorpusConfig,
        request: SourceJobRequest, limits: JobHostLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_sentiment_scheduled(model, planner, config, request, limits,
            None, Some(prefill), Some((kernel, compute_threads)), reader, cancellation)
    }
    /// Explicit compact retained classification with one nonrenewable
    /// invocation tokenizer allowance. Original policy and all four total
    /// caps authenticate on resume; physical scheduling grants no new work.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_classify_scheduled<R>(&self, model: &ResidentInt8, planner: Arc<ClassificationPlanner>,
        config: ClassificationCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        encoding_work_limits: Option<SafeBpeWorkLimits>, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let encoding = EncodingWorkScope::new(planner.document_encoding(), encoding_work_limits)
            .map_err(HostedError::BatchSetup)?;
        run(self, model, ClassificationTask { planner, config, encoding }, request, limits,
            prefill, team, reader, cancellation)
    }

    /// Retain complete independent-axis results with the same invocation
    /// tokenizer owner across original or retained-record job populations.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_sentiment_scheduled<R>(&self, model: &ResidentInt8, planner: Arc<SentimentPlanner>,
        config: SentimentCorpusConfig, request: SourceJobRequest, limits: JobHostLimits,
        encoding_work_limits: Option<SafeBpeWorkLimits>, prefill: Option<Int8PrefillLimits>,
        team: Option<(Int8Kernel, usize)>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let encoding = EncodingWorkScope::new(planner.document_encoding(), encoding_work_limits)
            .map_err(HostedError::BatchSetup)?;
        run(self, model, SentimentTask { planner, config, encoding }, request, limits,
            prefill, team, reader, cancellation)
    }
}

struct ClassificationTask {
    planner: Arc<ClassificationPlanner>, config: ClassificationCorpusConfig, encoding: Arc<EncodingWorkScope>,
}
impl Int8JobTask for ClassificationTask {
    fn identity(&self) -> &ExecutionIdentity { &self.config.identity }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        Some(Box::new((Arc::clone(&self.planner), Arc::clone(&self.encoding))))
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> {
        self.encoding.workspace_bytes().map_err(HostedError::BatchSetup)
    }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        validate_work(self.config.max_model_work)?;
        capacity(self.config.task_ceiling, native.kv_bytes, job.max_result_bytes)?;
        Ok(())
    }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let Self { planner, config, encoding } = self;
        let output_bytes = config.task_ceiling.max_output_bytes;
        let factory = Int8ClassificationJobPlanner::with_encoding_scope(&planner, config.identity,
            config.task_ceiling, config.planning, config.defaults, config.max_model_work, encoding)
            .map_err(HostedError::BatchSetup)?;
        let admission = ClassificationJobAdmission { output_bytes, inner: admission };
        let result = (|| {
            let processor = match prefill {
                None => Int8ClassificationJobProcessor::new(factory, engine, admission),
                Some(prefill) => Int8ClassificationJobProcessor::new_layer_major(factory, engine, admission, prefill),
            }.map_err(JobRunError::Processor)?;
            run_population(request, population, processor, control)
        })();
        Ok(result)
    }
}

struct SentimentTask {
    planner: Arc<SentimentPlanner>, config: SentimentCorpusConfig, encoding: Arc<EncodingWorkScope>,
}
impl Int8JobTask for SentimentTask {
    fn identity(&self) -> &ExecutionIdentity { &self.config.identity }
    fn retain_for_join(&self) -> Option<Box<dyn Send>> {
        Some(Box::new((Arc::clone(&self.planner), Arc::clone(&self.encoding))))
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> {
        self.encoding.workspace_bytes().map_err(HostedError::BatchSetup)
    }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        self.config.validate_with_document_encoding(&self.planner).map_err(HostedError::BatchSetup)?;
        capacity(self.config.task_ceiling, native.kv_bytes, job.max_result_bytes)?;
        Ok(())
    }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let Self { planner, config, encoding } = self;
        let factory = Int8SentimentJobPlanner::with_encoding_scope(&planner, config, encoding).map_err(HostedError::BatchSetup)?;
        let result = (|| {
            let processor = match prefill {
                None => Int8SentimentJobProcessor::new(factory, engine, admission),
                Some(prefill) => Int8SentimentJobProcessor::new_layer_major(factory, engine, admission, prefill),
            }.map_err(JobRunError::Processor)?;
            run_population(request, population, processor, control)
        })();
        Ok(result)
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
