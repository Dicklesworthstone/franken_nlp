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
        self.job_int8_extract_document_scheduled(model, planner, vocabulary, request, limits,
            reduction_reserve_bytes, None, None, reader, cancellation)
    }

    /// Share the admitted engine, prompt buffers and projection team across all
    /// chunks of each pending document. Only complete document maps are stored.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_extract_document_scheduled<R>(&self, model: &ResidentInt8,
        planner: Int8DocumentExtractionJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        let task = DocumentExtractionTask { planner, vocabulary, reduction_reserve_bytes,
            context_tokens: limits.native.context_tokens };
        scored::run(self, model, task, request, limits, prefill, team, reader, cancellation)
    }
}

struct DocumentExtractionTask {
    planner: Int8DocumentExtractionJobPlanner,
    vocabulary: Arc<ExtractionVocabulary>,
    reduction_reserve_bytes: u64,
    context_tokens: usize,
}
impl scored::Int8JobTask for DocumentExtractionTask {
    fn identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn validate(&self, native: Int8MemoryRequirement, job: JobLimits) -> Result<(), HostedError> {
        self.planner.check_job_limits(job).map_err(HostedError::BatchSetup)?;
        check_capacity(self.planner.document_budget(), self.planner.document_limits().mapping.chunks.context_tokens,
            self.context_tokens, native.kv_bytes, self.reduction_reserve_bytes)
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> {
        extraction_document_job_workspace(&self.planner, self.reduction_reserve_bytes)
    }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let result = Int8DocumentExtractionJobProcessor::scheduled(&self.planner, engine,
            &self.vocabulary, admission, prefill).map_err(JobRunError::Processor)
            .and_then(|processor| run_population(request, population, processor, control));
        Ok(result)
    }
}

/// Complete map, result staging and reducer storage carried alongside the job.
pub(crate) fn extraction_document_job_workspace(planner: &Int8DocumentExtractionJobPlanner, reserve: u64)
    -> Result<u64, HostedError> {
    let mapping = planner.document_limits();
    let object = match planner.root_object_reconciliation() {
        None => 0,
        Some(reconciliation) => {
            use crate::tasks::extract::object_reconcile::RootObjectSchema;
            let limits = crate::batch::extract::quantized::long::object::Int8ObjectExtractionMapLimits {
                mapping, reconciliation };
            sum(&[limits.reconciliation_workspace_bytes().map_err(HostedError::Int8ObjectExtractionMap)?,
                RootObjectSchema::workspace_bytes(planner.compiler_limits())
                    .map_err(|error| HostedError::Int8ObjectExtractionMap(error.into()))?])?
        }
    };
    sum(&[crate::hosted::extraction_map::document_workspace_bytes(planner.document_budget(), mapping,
        mapping.mapping.chunks.max_chunks, reserve)?,
        planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)?, object])
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
