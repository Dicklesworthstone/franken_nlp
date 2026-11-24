//! Whole-source document jobs on the same process-owned native invocation.
use super::*;
use crate::jobs::runner::source::document::{Int8SourceDocumentJobPlanner,
    Int8SourceDocumentJobProcessor, SourceDocumentAdmission, SourceDocumentBatchAdmission,
    reduction::Int8ReducedSourceDocumentJobProcessor};

impl NlpEngine {
    /// Create/resume original-document NER, keyphrase or cited-summary maps.
    /// An explicitly reduced factory selects whole-document summaries/rankings.
    /// All chunk identities and the complete output are admitted together. The
    /// same engine serves every pending document; no nested runtime or retry.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_source_document<R>(&self, model: &ResidentInt8, planner: Int8SourceDocumentJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest, limits: JobHostLimits,
        reduction_reserve_bytes: u64, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.job_int8_source_document_scheduled(model, planner, vocabulary, request, limits,
            reduction_reserve_bytes, None, None, reader, cancellation)
    }

    /// Execute all pending document maps or exact reducers with one physical
    /// schedule. Every chunk identity and complete result remains authenticated.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int8_source_document_scheduled<R>(&self, model: &ResidentInt8, planner: Int8SourceDocumentJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest, limits: JobHostLimits,
        reduction_reserve_bytes: u64, prefill: Option<Int8PrefillLimits>, team: Option<(Int8Kernel, usize)>,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        planner.check_job_limits(request.limits).map_err(JobRunError::Processor)?;
        let task = SourceDocumentTask { planner, vocabulary, reduction_reserve_bytes,
            context_tokens: limits.native.context_tokens };
        scored::run(self, model, task, request, limits, prefill, team, reader, cancellation)
    }
}

struct SourceDocumentTask {
    planner: Int8SourceDocumentJobPlanner,
    vocabulary: Arc<ExtractionVocabulary>,
    reduction_reserve_bytes: u64,
    context_tokens: usize,
}
impl scored::Int8JobTask for SourceDocumentTask {
    fn identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn validate(&self, native: Int8MemoryRequirement, _: JobLimits) -> Result<(), HostedError> {
        if native.kv_bytes == 0 || native.kv_bytes > self.planner.task_budget().max_kv_bytes
            || self.planner.planning_limits().max_context_tokens > self.context_tokens {
            return Err(HostedError::Limits("source document job complete native capacity"));
        }
        Ok(())
    }
    fn preparation_workspace_bytes(&self) -> Result<u64, HostedError> {
        source_document_job_workspace(&self.planner, self.reduction_reserve_bytes)
    }
    fn execute(self, engine: &mut StrictInt8Engine<'_>, admission: CorpusAdmission<'_>,
        request: SourceJobRequest, population: &Population<'_>, prefill: Option<Int8PrefillLimits>,
        control: &mut RunControl) -> Result<Result<JobProgress, JobRunError>, HostedError> {
        let admission = DocumentAdmissionHost { task: &self.planner.execution_identity().task_spec, inner: admission };
        let result = if self.planner.reduction_policy().is_some() {
            Int8ReducedSourceDocumentJobProcessor::scheduled(&self.planner, engine, &self.vocabulary,
                admission, prefill).map_err(JobRunError::Processor)
                .and_then(|processor| run_population(request, population, processor, control))
        } else {
            Int8SourceDocumentJobProcessor::scheduled(&self.planner, engine, &self.vocabulary,
                admission, prefill).map_err(JobRunError::Processor)
                .and_then(|processor| run_population(request, population, processor, control))
        };
        Ok(result)
    }
}

pub(crate) fn source_document_job_workspace(planner: &Int8SourceDocumentJobPlanner, reserve: u64) -> Result<u64, HostedError> {
    let mapping = planner.document_limits();
    let budget = planner.task_budget();
    if reserve == 0 { return Err(HostedError::Limits("source document reduction reservation")); }
    let chunks = mapping.chunks.max_chunks as u64;
    let native = budget.max_output_bytes.checked_mul(chunks)
        .ok_or(HostedError::Limits("source document result arithmetic"))?;
    let staged = sum(&[native, mapping.reduction.max_live_value_bytes as u64])?.checked_mul(4)
        .ok_or(HostedError::Limits("source document staging arithmetic"))?;
    let tokens = u64::from(budget.max_output_tokens).checked_mul(chunks).and_then(|n| n.checked_mul(8))
        .ok_or(HostedError::Limits("source document token arithmetic"))?;
    // In addition to JobHostLimits' explicit preparation/journal/IO/serializer
    // reservations. Price the maximum frontier, not just the first document.
    // These are modeled commitments, not allocator or OS-enforced RSS bounds.
    // The same cumulative encoder stays live across population preparation and
    // native reconstruction. Its transient workspace coexists with the entire
    // retained frontier; compact input never discounts original source storage.
    let encoding = planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)?;
    sum(&[staged, tokens, reserve, encoding])
}

// Do not widen extraction's admission implementation to accept other tasks.
// Reuse its all-identities contract with a distinct fixed-source-task host.
struct DocumentAdmissionHost<'a> { inner: CorpusAdmission<'a>, task: &'a str }
impl SourceDocumentBatchAdmission for DocumentAdmissionHost<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: SourceDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Pending), BatchItemFailure> {
        check_task_identities(self.task, request.identities)?;
        if request.mask_node_visits == 0 || request.mask_limits.max_trie_node_visits == 0
            || request.mask_limits.checkpoint_interval_nodes == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        for identity in request.identities {
            check_request(self.inner.model, self.inner.kv_bytes, 0, self.inner.output_bytes,
                identity, request.model_work, request.kv_reservation_bytes, 0, request.max_result_bytes)?;
        }
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        identities.extend(request.identities.iter().cloned());
        let tokens = request.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64;
        let guard = output_claim(self.inner.lease, request.max_result_bytes, tokens).map_err(admission_failure)?;
        Ok((identities, guard))
    }
}
fn check_task_identities(task: &str, identities: &[ExecutionIdentity]) -> Result<(), BatchItemFailure> {
    if !matches!(task, "ner-v1" | "keyphrases-v1" | "summarize-v1") || !(1..=256).contains(&identities.len()) {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    for identity in identities {
        identity.validate().map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != task { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    }
    Ok(())
}

#[cfg(test)] mod tests;
