//! Complete-document extraction with authenticated resume and physical ownership.
use super::*;
use crate::{jobs::runner::extract::document::{Int8DocumentExtractionJobPlanner,
    Int8DocumentExtractionJobProcessor}, tasks::ir::TaskBudget};

impl NlpEngine {
    /// Explicitly retain/resume whole-document results on one native invocation.
    /// One journal ordinal is one original document, never a partly finished
    /// chunk map. The original population is supplied again for authenticated
    /// resume; committed documents are not inferred again to rebuild output.
    /// Preparation must price the immutable recipe/config, its active adapter
    /// copy, and every simultaneously retained source/schema compilation plan.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_extract_document<R>(&self, model: &ResidentInt8,
        planner: Int8DocumentExtractionJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), planner.execution_identity())?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let required = requirements(limits.native)?;
        let mapping = planner.document_limits();
        check_capacity(planner.document_budget(), mapping.mapping.chunks.context_tokens,
            limits.native.context_tokens, required.kv_bytes, reduction_reserve_bytes)?;
        let bytes = sum(&[limits.reservation_bytes(request.limits)?,
            crate::hosted::extraction_map::document_workspace_bytes(planner.document_budget(), mapping,
                mapping.mapping.chunks.max_chunks, reduction_reserve_bytes)?])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input; // Complete charged owner, including queued cancellation.
            let (planner, vocabulary, request) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            // Existing document admission validates ALL child identities and
            // guards ONE complete document result through spool+journal commit.
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0,
                output_bytes: request.limits.max_result_bytes as u64 };
            let result = (|| {
                let processor = Int8DocumentExtractionJobProcessor::new(&planner, &mut engine.value,
                    &vocabulary, admission).map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            // Runner/journal/spool/lock and all result guards have drained.
            // Retain the factory and original population until native cleanup.
            drop(engine); drop(population); drop(vocabulary); drop(planner);
            drop(input); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}
fn check_capacity(budget: TaskBudget, planned_context: usize, resident_context: usize,
    kv: u64, reserve: u64) -> Result<(), HostedError> {
    if planned_context != resident_context || resident_context == 0
        || kv == 0 || kv > budget.max_kv_bytes || reserve == 0 {
        return Err(HostedError::Limits("document extraction job context, KV or reduction reservation"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_resident_capacity_and_reduction_headroom_are_mandatory() {
        let budget = TaskBudget { max_input_tokens: 1024, max_output_tokens: 32,
            max_output_bytes: 4096, max_grammar_states: 4096, max_kv_bytes: 8192 };
        check_capacity(budget, 128, 128, 8192, 4096).unwrap();
        for (planned, resident, kv, reserve) in [(127, 128, 8192, 4096), (128, 0, 8192, 4096),
            (128, 128, 0, 4096), (128, 128, 8193, 4096), (128, 128, 8192, 0)] {
            assert!(check_capacity(budget, planned, resident, kv, reserve).is_err());
        }
    }
    #[test]
    fn factory_and_owned_population_reader_cross_the_physical_boundary() {
        fn send<T: Send + 'static>() {}
        send::<Int8DocumentExtractionJobPlanner>();
        send::<StreamInput<(Int8DocumentExtractionJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest),
            std::io::Cursor<Vec<u8>>, ()>>();
    }
}
