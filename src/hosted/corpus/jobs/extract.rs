//! Authenticated schema-extraction jobs on one real hosted native invocation.
use super::*;
use crate::jobs::runner::extract::{Int8ExtractionJobPlanner, Int8ExtractionJobProcessor};

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
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        let bytes = limits.reservation_bytes(request.limits)?;
        let required = requirements(limits.native)?;
        check_model_identity(model.artifact_identity(), planner.execution_identity())?;
        check_storage(planner.task_ceiling(), request.limits, required.kv_bytes)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input; // Capture the complete storage-before-charge package.
            let (planner, vocabulary, request) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0, output_bytes: request.limits.max_result_bytes as u64 };
            let result = (|| {
                let processor = Int8ExtractionJobProcessor::new(planner, &mut engine.value, &vocabulary, admission)
                    .map_err(JobRunError::Processor)?;
                // The exact recipe/original population authenticates BEFORE
                // resume can repair a tail or admit any pending native work.
                run_population(request, &population, processor, control)
            })();
            // Journal/spool/lock and processor drop inside run_population.
            // Result guards have already survived spool sync + journal commit.
            drop(engine);
            drop(population);
            drop(vocabulary);
            drop(input);
            drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
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
