//! Authenticated resumable NER, keyphrases, cited summaries and passage QA
//! on the actual resident grouped-INT4 engine.
use super::*;
use crate::{
    batch::source::grouped_int4::{Int4SourceAdmission, Int4SourceBatchAdmission},
    jobs::runner::source::grouped_int4::{Int4SourceJobPlanner, Int4SourceJobProcessor},
};

struct SourceJobInput<R> {
    preparation: Option<(Int4SourceJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
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
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let native_work = planner.native_limits();
        // The factory prices its complete planning context into task_ceiling.
        // Equality here binds that context to the actual resident KV before
        // reading any population bytes or opening a job directory.
        let required = job_requirements(model.model_identity().group_width(), planner.task_ceiling(),
            request.limits, limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(SourceJobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
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
            let admission = SourceJobAdmission { inner: CorpusAdmission {
                lease: &lease, model: model.model_identity(), kv_bytes: required.native.kv_bytes,
                masks: native_work.masks, output_bytes: request.limits.max_result_bytes as u64,
            }, task_spec: &planner.execution_identity().task_spec };
            let result = (|| {
                let processor = Int4SourceJobProcessor::new(&planner, &mut engine.value, &vocabulary, admission)
                    .map_err(JobRunError::Processor)?;
                // JobRunner authenticates the complete population and recipe
                // before replay or tail repair. The native output guard stays
                // inside its result through spool sync and journal commit.
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            drop(population);
            drop(vocabulary);
            drop(planner);
            drop(input);
            drop(lease);
            Ok(result)
        })?;
        // Physical cancellation/drain failure takes precedence over a job
        // return value, including a durable commit with a lost acknowledgement.
        completed.map_err(HostedJobError::Job)
    }
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
        let _entry = NlpEngine::job_int4_source::<Cursor<Vec<u8>>>;
    }
}
