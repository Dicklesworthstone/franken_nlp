//! Complete-document INT4 pipelines inside the existing owned-job lifecycle.
//! One original document is one durable item, including all permitted passes.
use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    hosted::grouped_int4::source_map::Int4SourceMapConfig,
    jobs::runner::source::grouped_int4::document::{
        Int4SourceDocumentAdmission, Int4SourceDocumentBatchAdmission,
        Int4SourceDocumentJobPlanner, Int4SourceDocumentJobProcessor, Int4SourceDocumentTask,
    },
    tasks::source_planning::{SourcePlanningLimits, grouped_int4::long::Int4SourceMapLimits},
};

struct DocumentJobInput<R> {
    preparation: Option<(Int4SourceDocumentJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DocumentRequirements {
    native: Int4MemoryRequirement,
    native_workspace_bytes: u64,
    input_bytes: u64,
    document_workspace_bytes: u64,
}

impl Int4SourceDocumentJobPlanner {
    /// Quote the maximum complete document frontier, including every allowed
    /// synthesis/hierarchy pass and independent citation-verification storage.
    /// This reuses the single-document host's checked accounting with borrowed
    /// private settings. It is a workspace quote, not a memory permit or RSS
    /// claim; the job entrypoint also validates and reserves native, input and
    /// output storage before executing any document.
    pub fn workspace_bytes(&self, native: NativeLimits, reduction_reserve_bytes: u64)
        -> Result<u64, HostedError> {
        let chunks = self.document_limits().chunks.max_chunks;
        match self.task() {
            Int4SourceDocumentTask::Map(_) | Int4SourceDocumentTask::Summary { .. } =>
                quote_config(self, (), native, reduction_reserve_bytes).workspace_bytes(chunks),
            Int4SourceDocumentTask::Question(question) =>
                quote_config(self, question, native, reduction_reserve_bytes).question_workspace_bytes(chunks),
            Int4SourceDocumentTask::SummarySynthesis(request) =>
                quote_config(self, request, native, reduction_reserve_bytes).summary_synthesis_workspace_bytes(chunks),
            Int4SourceDocumentTask::SummaryHierarchy { request, limits } =>
                quote_config(self, request, native, reduction_reserve_bytes)
                    .summary_hierarchy_workspace_bytes(chunks, *limits),
            Int4SourceDocumentTask::QuestionSynthesis(request) =>
                quote_config(self, request, native, reduction_reserve_bytes).question_synthesis_workspace_bytes(chunks),
            Int4SourceDocumentTask::QuestionHierarchy { request, limits } =>
                quote_config(self, request, native, reduction_reserve_bytes)
                    .question_hierarchy_workspace_bytes(chunks, *limits),
        }
    }
}

fn quote_config<T>(planner: &Int4SourceDocumentJobPlanner, task: T,
    native: NativeLimits, reduction_reserve_bytes: u64) -> Int4SourceMapConfig<T> {
    Int4SourceMapConfig {
        identity: planner.execution_identity().clone(), task, budget: planner.task_budget(),
        planning: planner.planning_limits(), mapping: planner.document_limits(), native,
        // The workspace-only methods do not inspect or grant preparation
        // memory. The job host separately prices its real preparation owner.
        preparation_reserve_bytes: 0, reduction_reserve_bytes,
    }
}

impl NlpEngine {
    /// Run or resume complete original documents through the existing INT4
    /// map, summary-selection, cited-synthesis and hierarchical QA pipelines.
    /// Each document commits one complete result; intermediate chunks/passes
    /// are never exposed as separately resumable or successful partial items.
    ///
    /// The immutable model-bound factory and entire original population are
    /// authenticated by JobRunner before tail repair or native execution.
    /// All native work, masks, verification and reduction allowances remain
    /// nonrenewable across failures and resumes. The output guard survives
    /// spool sync and journal acknowledgement. Private questions, inputs and
    /// results are not emitted to stdout by this method.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source_document<R>(&self, model: &ResidentInt4,
        planner: Int4SourceDocumentJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, reduction_reserve_bytes: u64,
        kernel: PackedInt4Kernel, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let mapping = planner.document_limits();
        let result_bytes = planner.max_result_bytes();
        let output_tokens = planner.output_token_ceiling().map_err(HostedError::BatchSetup)?;
        let document_workspace = planner.workspace_bytes(limits.native, reduction_reserve_bytes)?;
        let required = document_requirements(model.model_identity().group_width(),
            planner.task_budget(), planner.planning_limits(), mapping, result_bytes,
            request.limits, limits, document_workspace, private_question_bytes(planner.task()))?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(DocumentJobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        // The maximum frontier remains reserved even when an early document
        // is short. Later documents and hierarchy levels cannot borrow from
        // uncharged preparation or from another concurrent request.
        let frontier = Pending::reserve(&lease, MemoryClass::JobBuffers, required.document_workspace_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.native_workspace_bytes)?;
        let model = model.clone();
        let completed = dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let frontier = frontier;
            checkpoint(control)?;
            let (planner, vocabulary, request) = input.value.preparation.take()
                .ok_or(HostedError::CompletionMissing)?;
            let population = match JobPopulation::read_ndjson(&mut input.value.reader,
                &request.key, request.job_id, request.limits, limits.transport, control) {
                Ok(population) => population,
                Err(error) => return Ok(Err(JobRunError::Storage(error))),
            };
            checkpoint(control)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                        max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = DocumentAdmission {
                lease: &lease, model: model.model_identity(),
                task_spec: &planner.execution_identity().task_spec,
                kv_bytes: required.native.kv_bytes, mapping, result_bytes, output_tokens,
            };
            let result = (|| {
                let processor = Int4SourceDocumentJobProcessor::new(
                    &planner, &mut engine.value, &vocabulary, admission)
                    .map_err(JobRunError::Processor)?;
                run_population(request, &population, processor, control)
            })();
            drop(engine);
            drop(population);
            drop(vocabulary);
            drop(planner);
            drop(input);
            drop(frontier);
            drop(lease);
            Ok(result)
        })?;
        // A physical cancellation/drain failure takes precedence over the job
        // result. Authenticated resume reconciles a lost acknowledgement.
        completed.map_err(HostedJobError::Job)
    }
}

fn private_question_bytes(task: &Int4SourceDocumentTask) -> u64 {
    match task {
        Int4SourceDocumentTask::Question(question) => question.question.capacity() as u64,
        Int4SourceDocumentTask::QuestionSynthesis(request)
        | Int4SourceDocumentTask::QuestionHierarchy { request, .. } =>
            request.question.question.capacity() as u64,
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn document_requirements(width: usize, budget: TaskBudget, planning: SourcePlanningLimits,
    mapping: Int4SourceMapLimits, result_bytes: u64, job: JobLimits, limits: JobHostLimits,
    document_workspace_bytes: u64, private_question_bytes: u64)
    -> Result<DocumentRequirements, HostedJobError> {
    budget.validate().map_err(|_| HostedError::Limits("INT4 document job task budget"))?;
    let input_bytes = sum(&[limits.required_buffer_bytes(job)?, private_question_bytes])?;
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width)
        .map_err(native_error)?;
    if limits.native.allocator_reserve_bytes == 0 || document_workspace_bytes == 0
        || planning.max_context_tokens != limits.native.context_tokens
        || native.kv_bytes != budget.max_kv_bytes
        || !(1..=256).contains(&mapping.chunks.max_chunks)
        || result_bytes == 0 || result_bytes > job.max_result_bytes as u64
        || !work_fits(mapping.max_model_work, job.max_work.int4_model_work())
        || mapping.max_mask_visits > job.max_work.mask_node_visits {
        return Err(HostedError::Limits("INT4 document job complete resident and lifetime envelope").into());
    }
    let native_workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    Ok(DocumentRequirements { native, native_workspace_bytes, input_bytes, document_workspace_bytes })
}

struct DocumentAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    task_spec: &'a str,
    kv_bytes: u64,
    mapping: Int4SourceMapLimits,
    result_bytes: u64,
    output_tokens: u64,
}
impl Int4SourceDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4SourceDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        check_document_request(self.model.group_width(), self.task_spec, self.kv_bytes,
            self.mapping, self.result_bytes, self.output_tokens, &request)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        for identity in request.identities {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            identities.push(identity.clone());
        }
        let guard = output_claim(self.lease, request.max_result_bytes, request.max_output_tokens)
            .map_err(admission_failure)?;
        Ok((identities, guard))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_document_request(width: usize, task_spec: &str, kv_bytes: u64,
    mapping: Int4SourceMapLimits, result_bytes: u64, output_tokens: u64,
    request: &Int4SourceDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    if !matches!(task_spec, "ner-v1" | "keyphrases-v1" | "summarize-v1" | "answer-v1")
        || request.chunk_count == 0 || request.chunk_count > mapping.chunks.max_chunks
        || request.chunk_count > 256 || request.identities.is_empty()
        || request.identities.len() > request.chunk_count
        || request.kv_reservation_bytes != kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    for identity in request.identities {
        constrained_int4::check_profile(identity, width)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != task_spec {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
    }
    if request.mask_node_visits == 0 || request.mask_node_visits > mapping.max_mask_visits
        || !same_masks(request.mask_limits, mapping.mask_limits) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let work = request.model_work;
    if work.forward_positions == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || work.projected_logits == 0 || work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || work.projected_logits / NANBEIGE_VOCAB_SIZE as u64 > work.forward_positions
        || !work_fits(work, mapping.max_model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    // Work is the sum of real independent sequences and reserved future
    // passes. It must not be reinterpreted as one context-sized sequence or
    // replaced with the similarly shaped INT8 projection formula.
    if request.max_result_bytes == 0 || request.max_result_bytes != result_bytes
        || request.max_output_tokens > output_tokens {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}

fn same_masks(left: MaskWorkLimits, right: MaskWorkLimits) -> bool {
    left.max_trie_node_visits == right.max_trie_node_visits
        && left.checkpoint_interval_nodes == right.checkpoint_interval_nodes
}
fn work_fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions
        && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs
        && work.projections.dot_products <= ceiling.projections.dot_products
        && work.projections.multiply_accumulates <= ceiling.projections.multiply_accumulates
}

#[cfg(test)]
mod tests;
