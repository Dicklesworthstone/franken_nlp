//! Durable exact-schema extraction for complete original INT4 documents.
//! The existing native map and owned-job runner retain their original contracts.
use super::*;
use crate::{
    batch::extract::grouped_int4::long::Int4ExtractionMapLimits,
    grammar::mask::MaskWorkLimits,
    hosted::grouped_int4::extraction_map::document_workspace_bytes,
    jobs::runner::extract::grouped_int4::document::{Int4DocumentExtractionJobPlanner,
        Int4DocumentExtractionJobProcessor, Int4ExtractionDocumentAdmission,
        Int4ExtractionDocumentBatchAdmission},
};

struct ExtractionDocumentInput<R> {
    preparation: Option<(Int4DocumentExtractionJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DocumentRequirements {
    native: Int4MemoryRequirement,
    native_workspace_bytes: u64,
    input_bytes: u64,
    document_workspace_bytes: u64,
}

impl Int4DocumentExtractionJobPlanner {
    /// Checked maximum document-frontier quote using the same calculation as
    /// `extract_int4_document`. Native, private preparation, population and
    /// final-result ownership are reserved separately by the job host.
    pub fn workspace_bytes(&self, native: NativeLimits, reduction_reserve_bytes: u64)
        -> Result<u64, HostedError> {
        let document = self.document_limits();
        if native.context_tokens != document.mapping.chunks.context_tokens {
            return Err(HostedError::Limits("INT4 extraction document job resident context"));
        }
        document_workspace_bytes(self.document_budget(), document,
            document.mapping.chunks.max_chunks, reduction_reserve_bytes)
    }
}

impl NlpEngine {
    /// Start or resume fixed-schema extraction over the complete original
    /// population. Every document commits all independently validated chunk
    /// values and source locations as one result, with no partial chunk commit
    /// or invented global JSON merge. A required field that cannot be produced
    /// under its schema/source constraint remains a typed no-result failure.
    ///
    /// The immutable factory and complete input population authenticate before
    /// native execution or tail repair. Exact group-aware work and masks are
    /// debited before execution; failed attempts and resumes do not replenish
    /// model or verification authority. Output guards survive spool sync and
    /// journal acknowledgement. This API does not publish private results to
    /// stdout or retain an input spool for future resume.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_extract_document<R>(&self, model: &ResidentInt4,
        planner: Int4DocumentExtractionJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
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
        if planner.group_width() != model.model_identity().group_width()
            || planner.execution_identity().task_spec != "extract-v1" {
            return Err(HostedError::ModelIdentity.into());
        }
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let document = planner.document_limits();
        let budget = planner.document_budget();
        let result_bytes = planner.max_result_bytes();
        let frontier_bytes = planner.workspace_bytes(limits.native, reduction_reserve_bytes)?;
        let required = requirements(planner.group_width(), budget, document, result_bytes,
            request.limits, limits, frontier_bytes, planner.request().schema.capacity())?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(ExtractionDocumentInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        // Reserve the maximum complete frontier before population ingestion;
        // a later long document cannot exceed an early short document's claim.
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
            let admission = DocumentAdmission { lease: &lease, model: model.model_identity(),
                base: planner.execution_identity(), kv_bytes: required.native.kv_bytes,
                document, result_bytes, output_tokens_per_chunk: u64::from(budget.max_output_tokens) };
            let result = (|| {
                let processor = Int4DocumentExtractionJobProcessor::new(
                    &planner, &mut engine.value, &vocabulary, admission).map_err(JobRunError::Processor)?;
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
        // Cancellation/drain failure takes precedence. Authenticated resume
        // reconciles durable progress if a commit acknowledgement was lost.
        completed.map_err(HostedJobError::Job)
    }
}

#[allow(clippy::too_many_arguments)]
fn requirements(width: usize, budget: TaskBudget, document: Int4ExtractionMapLimits,
    result_bytes: u64, job: JobLimits, limits: JobHostLimits, document_workspace_bytes: u64,
    schema_capacity: usize) -> Result<DocumentRequirements, HostedJobError> {
    budget.validate().map_err(|_| HostedError::Limits("INT4 extraction document job task budget"))?;
    // The frozen private recipe and native shared-default compiler each own
    // schema bytes. Their variable capacity is additional to explicit planner
    // overhead and all population/serializer/journal buffers.
    let schemas = (schema_capacity as u64).checked_mul(2)
        .ok_or(HostedError::Limits("INT4 extraction document job schema arithmetic"))?;
    let input_bytes = sum(&[limits.required_buffer_bytes(job)?, schemas])?;
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    let mapping = document.mapping;
    if schema_capacity == 0 || document_workspace_bytes == 0 || limits.native.allocator_reserve_bytes == 0
        || mapping.chunks.context_tokens != limits.native.context_tokens
        || native.kv_bytes != budget.max_kv_bytes || !(1..=256).contains(&mapping.chunks.max_chunks)
        || result_bytes == 0 || result_bytes > job.max_result_bytes as u64
        || !crate::jobs::JobWork::from_int4(mapping.max_model_work, mapping.max_mask_visits).fits(job.max_work) {
        return Err(HostedError::Limits("INT4 extraction document job complete resident and lifetime envelope").into());
    }
    let native_workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    Ok(DocumentRequirements { native, native_workspace_bytes, input_bytes, document_workspace_bytes })
}

struct DocumentAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    base: &'a ExecutionIdentity,
    kv_bytes: u64,
    document: Int4ExtractionMapLimits,
    result_bytes: u64,
    output_tokens_per_chunk: u64,
}
impl Int4ExtractionDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4ExtractionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        check_request(self.model.group_width(), self.base, self.kv_bytes, self.document,
            self.result_bytes, self.output_tokens_per_chunk, &request)?;
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
fn check_request(width: usize, base: &ExecutionIdentity, kv_bytes: u64, document: Int4ExtractionMapLimits,
    result_bytes: u64, output_tokens_per_chunk: u64, request: &Int4ExtractionDocumentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let mapping = document.mapping;
    if base.task_spec != "extract-v1" || request.chunk_count == 0
        || request.chunk_count > mapping.chunks.max_chunks || request.chunk_count > 256
        || request.identities.len() != request.chunk_count || request.kv_reservation_bytes != kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    for identity in request.identities {
        constrained_int4::check_profile(identity, width)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != "extract-v1" || identity.tokenizer_digest != base.tokenizer_digest
            || identity.template_digest != base.template_digest {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
    }
    let masks = mapping.mask_visits_per_chunk.checked_mul(request.chunk_count as u64);
    if masks != Some(request.mask_node_visits) || request.mask_node_visits == 0
        || request.mask_node_visits > mapping.max_mask_visits
        || !same_masks(request.mask_limits, mapping.mask_limits) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let work = request.model_work;
    if work.forward_positions == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || work.projected_logits == 0 || work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || work.projected_logits / NANBEIGE_VOCAB_SIZE as u64 > work.forward_positions
        || !crate::jobs::JobWork::from_int4(work, request.mask_node_visits)
            .fits(crate::jobs::JobWork::from_int4(mapping.max_model_work, mapping.max_mask_visits)) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    // Aggregate work consists of independent context-sized native sequences.
    // Preserve their actual INT4 group costs; do not reinterpret their sum as
    // one context or recompute it through the historical INT8 wire counters.
    if request.max_result_bytes == 0 || request.max_result_bytes != result_bytes
        || output_tokens_per_chunk.checked_mul(request.chunk_count as u64) != Some(request.max_output_tokens) {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}
fn same_masks(left: MaskWorkLimits, right: MaskWorkLimits) -> bool {
    left.max_trie_node_visits == right.max_trie_node_visits
        && left.checkpoint_interval_nodes == right.checkpoint_interval_nodes
}

#[cfg(test)]
mod tests;
