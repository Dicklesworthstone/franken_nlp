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
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        planner.check_job_limits(request.limits).map_err(JobRunError::Processor)?;
        check_model_identity(model.artifact_identity(), planner.execution_identity())?;
        let required = requirements(limits.native)?;
        if required.kv_bytes == 0 || required.kv_bytes > planner.task_budget().max_kv_bytes
            || planner.planning_limits().max_context_tokens > limits.native.context_tokens {
            return Err(HostedError::Limits("source document job complete native capacity").into());
        }
        let bytes = sum(&[limits.reservation_bytes(request.limits)?,
            document_workspace(&planner, reduction_reserve_bytes)?])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(StreamInput { planner: Some((planner, vocabulary, request)), reader, writer: () }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Whole aggregate capture also protects queued-cancellation drops.
            let mut input = input;
            let (planner, vocabulary, request) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader, &request.key,
                request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = DocumentAdmissionHost { task: &planner.execution_identity().task_spec,
                inner: CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                    kv_bytes: required.kv_bytes, sampler_bytes: 0, output_bytes: request.limits.max_result_bytes as u64 } };
            let result = (|| {
                if planner.reduction_policy().is_some() {
                    let processor = Int8ReducedSourceDocumentJobProcessor::new(&planner, &mut engine.value,
                        &vocabulary, admission).map_err(JobRunError::Processor)?;
                    run_population(request, &population, processor, control)
                } else {
                    let processor = Int8SourceDocumentJobProcessor::new(&planner, &mut engine.value,
                        &vocabulary, admission).map_err(JobRunError::Processor)?;
                    run_population(request, &population, processor, control)
                }
            })();
            // Runner/journal/spool and guarded results finish inside the call.
            // Native and retained preparation storage drop BEFORE their charge.
            drop(engine); drop(population); drop(vocabulary); drop(planner);
            drop(input); drop(lease);
            Ok(result)
        })?;
        completed.map_err(HostedJobError::Job)
    }
}

fn document_workspace(planner: &Int8SourceDocumentJobPlanner, reserve: u64) -> Result<u64, HostedError> {
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
    sum(&[staged, tokens, reserve])
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
