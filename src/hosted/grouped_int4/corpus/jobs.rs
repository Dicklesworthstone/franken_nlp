//! Authenticated resumable extraction on the charged grouped-INT4 host.
//! The existing runner owns retention, work debits and publication ordering.
use super::*;
mod source;
mod document;
use crate::{
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, run_population},
    jobs::{JobLimits, JobProgress, population::JobPopulation,
        runner::{JobRunError, extract::grouped_int4::{Int4ExtractionJobPlanner, Int4ExtractionJobProcessor}}},
    tasks::ir::TaskBudget,
};

struct JobInput<R> {
    preparation: Option<(Int4ExtractionJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}

impl NlpEngine {
    /// Create or resume an explicitly retained INT4 extraction job using the
    /// complete original NDJSON population and one exact shared schema.
    ///
    /// The model's strict-v2/group identity and all six lifetime work counters
    /// are checked before any input or job-file access. Population ingestion,
    /// authenticated resume, native execution, spool/journal commits and optional
    /// materialization complete inside one owned blocking invocation. Stored
    /// results remain protected and no result text is written to stdout here.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_extract<R>(&self, model: &ResidentInt4, planner: Int4ExtractionJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let native_work = planner.native_limits();
        let required = job_requirements(model.model_identity().group_width(), planner.task_ceiling(),
            request.limits, limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(JobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input; // Preserve storage-before-charge drop order.
            checkpoint(control)?;
            let (planner, vocabulary, request) = input.value.preparation.take()
                .ok_or(HostedError::CompletionMissing)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            let mut engine = allocate_native(kv, workspace, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = CorpusAdmission { lease: &lease, model: model.model_identity(),
                kv_bytes: required.native.kv_bytes, masks: native_work.masks,
                output_bytes: request.limits.max_result_bytes as u64 };
            let result = (|| {
                let processor = Int4ExtractionJobProcessor::new(planner, &mut engine.value, &vocabulary, admission)
                    .map_err(JobRunError::Processor)?;
                // Authentication precedes tail repair and any native forward.
                // Output guards survive spool sync and journal acknowledgement.
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            drop(population);
            drop(vocabulary);
            drop(input);
            drop(lease);
            Ok(result)
        })?;
        // Runtime cancellation/drain failure takes precedence over job return.
        completed.map_err(HostedJobError::Job)
    }
}

fn job_requirements(width: usize, task: TaskBudget, job: JobLimits, limits: JobHostLimits)
    -> Result<Requirements, HostedJobError> {
    let input_bytes = limits.required_buffer_bytes(job)?;
    if limits.native.allocator_reserve_bytes == 0 {
        return Err(HostedError::Limits("INT4 job allocator reserve").into());
    }
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    if native.kv_bytes == 0 || task.max_kv_bytes != native.kv_bytes
        || task.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("INT4 job complete KV and stored-result envelope").into());
    }
    let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    Ok(Requirements { native, workspace_bytes, input_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hosted::{NativeLimits, RunLimits};
    use crate::jobs::{JobWork, population::PopulationReadLimits};
    use std::time::Duration;

    fn fixture() -> (TaskBudget, JobLimits, JobHostLimits) {
        let native = Int4MemoryRequirement::for_context(64, 128).unwrap();
        let task = TaskBudget { max_input_tokens: 48, max_output_tokens: 16,
            max_output_bytes: 4096, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
        let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
            max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
            max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
            max_work: JobWork::default() };
        let host = JobHostLimits {
            native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 65536,
                run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000,
                    cleanup_reserve_bytes: 65536 } },
            transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
            preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 131072,
            journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
        };
        (task, job, host)
    }
    #[test]
    fn whole_population_native_workspace_and_full_context_are_priced_together() {
        let (task, job, host) = fixture();
        let needed = job_requirements(128, task, job, host).unwrap();
        assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap());
        assert_eq!(needed.native.kv_bytes, task.max_kv_bytes);
        assert_eq!(needed.workspace_bytes, needed.native.rope_bytes
            + needed.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    }
    #[test]
    fn looser_or_tighter_kv_and_oversized_results_refuse_before_input() {
        let (task, job, host) = fixture();
        for kv in [0, task.max_kv_bytes - 1, task.max_kv_bytes + 1] {
            let mut wrong = task; wrong.max_kv_bytes = kv;
            assert!(job_requirements(128, wrong, job, host).is_err());
        }
        let mut wrong = task; wrong.max_output_bytes = job.max_result_bytes as u64 + 1;
        assert!(job_requirements(128, wrong, job, host).is_err());
        let mut wrong = host; wrong.native.allocator_reserve_bytes = 0;
        assert!(job_requirements(128, task, job, wrong).is_err());
    }
    #[test]
    fn group_width_and_explicit_host_reserves_remain_mandatory() {
        let (task, job, host) = fixture();
        assert!(job_requirements(0, task, job, host).is_err());
        for axis in 0..4 {
            let mut wrong = host;
            match axis {
                0 => wrong.preparation_reserve_bytes = 0,
                1 => wrong.io_reserve_bytes = 0,
                2 => wrong.journal_reserve_bytes = 0,
                _ => wrong.serialization_reserve_bytes = 0,
            }
            assert!(job_requirements(128, task, job, wrong).is_err());
        }
    }
    #[test]
    fn owned_population_factory_secret_and_vocabulary_cross_one_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<Int4ExtractionJobPlanner>();
        send::<JobInput<std::io::Cursor<Vec<u8>>>>();
    }
}
