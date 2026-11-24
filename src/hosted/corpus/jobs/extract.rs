//! Authenticated schema-extraction jobs on one real hosted native invocation.
use super::*;
use crate::jobs::runner::extract::{Int8ExtractionJobPlanner, Int8ExtractionJobProcessor};
mod document;

impl NlpEngine {
    /// Create/resume raw-schema extraction with the same owner-only retention
    /// request, population freeze, durable debits and publication transaction as
    /// source jobs. Schemas remain exact strings, including decimal constants.
    /// Neither a parse failure nor a failed source check becomes a stored value.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_extract<R>(&self, model: &ResidentInt8, planner: Int8ExtractionJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest,
        limits: JobHostLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_extract_scheduled(model, planner, vocabulary, request, limits,
            None, None, reader, cancellation)
    }

    /// Run the authenticated schema job with optional bounded prompt morsels
    /// and one persistent projection team. The planner retains its exact
    /// schema, encoding ledger and resume commitment across scheduling choices.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_extract_scheduled<R>(&self, model: &ResidentInt8, planner: Int8ExtractionJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest, limits: JobHostLimits,
        prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        scored::run(self, model, ExtractionJobTask { planner, vocabulary }, request, limits,
            prefill, team, reader, cancellation)
    }
}

struct ExtractionJobTask {
    planner: Int8ExtractionJobPlanner,
    vocabulary: Arc<ExtractionVocabulary>,
}

impl scored::Int8JobTask for ExtractionJobTask {
    fn identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        check_storage(self.planner.task_ceiling(), job, native.kv_bytes)
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> {
        self.planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)
    }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let result = Int8ExtractionJobProcessor::scheduled(self.planner, engine, &self.vocabulary,
            admission, prefill).map_err(JobRunError::Processor)
            .and_then(|processor| run_population(request, population, processor, control));
        Ok(result)
    }
}

fn check_storage(task: crate::tasks::ir::TaskBudget, job: JobLimits, kv: u64) -> Result<(), HostedError> {
    if kv == 0 || kv > task.max_kv_bytes || task.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("extraction job whole KV and stored-result envelope"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_context_and_result_storage_are_admitted_not_only_the_prompt() {
        let task = crate::tasks::ir::TaskBudget { max_input_tokens: 1024, max_output_tokens: 32,
            max_output_bytes: 4096, max_grammar_states: 4096, max_kv_bytes: 8192 };
        let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
            max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
            max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
            max_work: crate::jobs::JobWork::default() };
        check_storage(task, job, 8192).unwrap();
        assert!(check_storage(task, job, 8193).is_err());
        assert!(check_storage(task, job, 0).is_err());
        let mut larger = task; larger.max_output_bytes = job.max_result_bytes as u64 + 1;
        assert!(check_storage(larger, job, 8192).is_err());
    }
    #[test]
    fn sealed_factory_and_owned_input_cross_the_actual_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<Int8ExtractionJobPlanner>();
        send::<StreamInput<(Int8ExtractionJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest),
            std::io::Cursor<Vec<u8>>, ()>>();
    }
}
