//! Complete original documents and private per-record questions on one stream.
use super::*;
use super::documents::admission::check_encoding_preparation;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        source::grouped_int4::question::{Int4QuestionDocumentBatchConfig,
            Int4QuestionDocumentBatchPlanner, NativeInt4QuestionDocumentBatch,
            Int4QuestionDocumentAdmission, Int4QuestionDocumentBatchAdmission,
            memory::{Int4QuestionDocumentMemoryLimits, Int4QuestionDocumentMemoryRequirement}}},
    hosted::corpus::CorpusLimits,
    native_engine::strict_int4::Int4Work,
    tasks::{answer::ANSWER_TASK_VERSION,
        source_planning::grouped_int4::long::cohort::Int4SourceMapCohortBudget},
};
#[cfg(all(feature = "metadata-store", target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod job;

struct QuestionStreamInput<R, W> {
    preparation: Option<(Int4QuestionDocumentBatchPlanner, Arc<ExtractionVocabulary>)>,
    reader: R,
    writer: W,
}

/// Pure, shared pre-IO quotation of transport, question copies, complete source
/// frontiers, citations, optional synthesis, native phases and output staging.
/// Caller-owned compiler/vocabulary and IO allocations must also fit their
/// explicit preparation and IO declarations. These are modeled reservations.
pub fn int4_question_document_corpus_requirements(width: usize,
    config: Int4QuestionDocumentBatchConfig, limits: CorpusLimits,
    reduction_reserve_bytes: u64, cohort_chunks: Option<usize>, hierarchy_groups: Option<usize>)
    -> Result<Int4QuestionDocumentMemoryRequirement, HostedError> {
    limits.native.run.validate()?;
    memory_limits(limits, reduction_reserve_bytes, cohort_chunks, hierarchy_groups)
        .requirements(width, config, limits.transport).map_err(map_error)
}
fn memory_limits(limits: CorpusLimits, reduction_reserve_bytes: u64, cohort_chunks: Option<usize>,
    hierarchy_groups: Option<usize>)
    -> Int4QuestionDocumentMemoryLimits {
    Int4QuestionDocumentMemoryLimits { context_tokens: limits.native.context_tokens,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, reduction_reserve_bytes, cohort_chunks, hierarchy_groups }
}

impl NlpEngine {
    /// Answer a different private question over each complete source record.
    /// All independent passage answers remain in the result. Explicit flat
    /// synthesis uses all collected, verified verbatim quotations and refuses
    /// evidence that cannot fit the declared continuation; it never truncates.
    /// Explicit hierarchy instead selects original quotations across bounded
    /// levels, retains every native pass and never claims lossless equivalence.
    ///
    /// One resident model, owned reader/writer, deadline and nonrenewable native,
    /// mask and verification ledgers span all records and flush epochs. Every
    /// discovery and optional continuation completes before an item is emitted.
    /// A failed chunk or continuation publishes no partial document result.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_question_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4QuestionDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let config = *planner.config();
        if planner.group_width() != model.model_identity().group_width()
            || planner.base_execution_identity().task_spec != ANSWER_TASK_VERSION
            || config.planning.max_context_tokens != limits.native.context_tokens {
            return Err(HostedError::ModelIdentity);
        }
        check_encoding_preparation(planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)?,
            limits.preparation_reserve_bytes)?;
        let cohort_chunks = planner.uses_chunk_cohorts().then_some(config.mapping.reduction.map_batch_chunks);
        let hierarchy_groups = planner.hierarchy_cohort_schedule().map(|schedule| schedule.max_groups);
        let required = int4_question_document_corpus_requirements(planner.group_width(), config,
            limits, reduction_reserve_bytes, cohort_chunks, hierarchy_groups)?;
        let cohort = planner.uses_native_cohorts().then(|| planner.cohort_requirements()
            .map_err(HostedError::BatchSetup)).transpose()?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(QuestionStreamInput { preparation: Some((planner, vocabulary)), reader, writer }))?;
        let frontier = Pending::reserve(&lease, MemoryClass::JobBuffers, required.document_workspace_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let (planner, vocabulary) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let admission = QuestionAdmission::new(&lease, model.model_identity(),
                planner.base_execution_identity(), config, required);
            let result = if let Some(quote) = cohort {
                let budget = Int4SourceMapCohortBudget { context_limit: limits.native.context_tokens,
                    native_memory: Int4MemoryBudget { max_kv_bytes: quote.native.kv_bytes,
                        max_rope_bytes: quote.native.rope_bytes,
                        max_scratch_payload_bytes: quote.native.scratch_payload_bound },
                    max_workspace_bytes: quote.workspace_bytes, max_result_bytes: quote.max_result_bytes };
                let result = {
                    let mut processor = NativeInt4QuestionDocumentBatch::new_cohorts(&planner,
                        &model.inner.loaded.value, &vocabulary, budget, kernel, admission)
                        .map_err(HostedError::BatchSetup)?;
                    stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)
                };
                // Exact-sized discovery and hierarchy windows (or their scalar
                // phases) drain sequentially under the same complete peak.
                // No unused scalar engine is allocated beside those owners.
                drop(scratch); drop(kv);
                result
            } else {
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound,
                    }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
                })?;
                let result = {
                    let mut processor = NativeInt4QuestionDocumentBatch::new(&planner, &mut engine.value,
                        &vocabulary, admission).map_err(HostedError::BatchSetup)?;
                    stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)
                };
                drop(engine);
                result
            };
            drop(vocabulary); drop(planner); drop(input);
            drop(frontier); drop(lease);
            result
        })
    }
}

#[derive(Clone, Copy)]
struct QuestionCapacity {
    config: Int4QuestionDocumentBatchConfig,
    kv_bytes: u64,
    max_result_bytes: u64,
    max_output_tokens: u64,
}
struct QuestionAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    base: &'a ExecutionIdentity,
    capacity: QuestionCapacity,
}
impl<'a> QuestionAdmission<'a> {
    fn new(lease: &'a EngineLease, model: &'a Int4ModelIdentity, base: &'a ExecutionIdentity,
        config: Int4QuestionDocumentBatchConfig, required: Int4QuestionDocumentMemoryRequirement) -> Self {
        Self { lease, model, base, capacity: QuestionCapacity { config, kv_bytes: required.native.kv_bytes,
            max_result_bytes: required.max_output_bytes, max_output_tokens: required.max_output_tokens } }
    }
}
impl Int4QuestionDocumentBatchAdmission for QuestionAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4QuestionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        check_request(self.model.group_width(), self.base, self.capacity, &request)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for identity in request.identities {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
            identities.push(identity.clone());
        }
        // Hierarchies retain native transcripts after their engines drain;
        // keep their exact declared token authority charged through delivery.
        let guard = output_claim(self.lease, request.max_result_bytes, request.max_output_tokens)
            .map_err(|error| match error { HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
                _ => BatchItemFailure::fatal(BatchCode::InvalidExecution) })?;
        Ok((identities, guard))
    }
}
fn check_request(width: usize, base: &ExecutionIdentity, capacity: QuestionCapacity,
    request: &Int4QuestionDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    let config = capacity.config; let mapping = config.mapping;
    if base.task_spec != ANSWER_TASK_VERSION || request.chunk_count == 0
        || request.chunk_count > mapping.chunks.max_chunks || request.chunk_count > 256
        || request.native_chunks == 0 || request.native_chunks > request.chunk_count
        || request.identities.len() != request.native_chunks || capacity.kv_bytes == 0
        || request.kv_reservation_bytes != capacity.kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    for identity in request.identities {
        constrained_int4::check_profile(identity, width)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != ANSWER_TASK_VERSION || identity.tokenizer_digest != base.tokenizer_digest
            || identity.template_digest != base.template_digest {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
    }
    let contexts = (request.native_chunks as u64).checked_add(config.continuation_passes() as u64);
    if contexts.and_then(|n| mapping.mask_visits_per_chunk.checked_mul(n)) != Some(request.mask_node_visits)
        || request.mask_node_visits == 0 || request.mask_node_visits > mapping.max_mask_visits
        || request.mask_limits.max_trie_node_visits != mapping.mask_limits.max_trie_node_visits
        || request.mask_limits.checkpoint_interval_nodes != mapping.mask_limits.checkpoint_interval_nodes {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    // These are independent contexts plus a possible reserved continuation,
    // never one concatenated sequence or a per-row allowance times the width.
    let work = request.model_work;
    if [work.forward_positions, work.projected_logits, work.attention_pairs,
        work.projections.dot_products, work.projections.multiply_accumulates].contains(&0)
        || work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || work.projected_logits / NANBEIGE_VOCAB_SIZE as u64 > work.forward_positions
        || !work_fits(work, mapping.max_model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if capacity.max_result_bytes == 0 || capacity.max_result_bytes != mapping.reduction.max_result_bytes as u64
        || request.max_result_bytes != capacity.max_result_bytes
        || config.retained_output_tokens().map_err(BatchItemFailure::fatal)? != capacity.max_output_tokens
        || request.max_output_tokens != capacity.max_output_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    Ok(())
}
fn work_fits(work: Int4Work, maximum: Int4Work) -> bool {
    work.forward_positions <= maximum.forward_positions && work.projected_logits <= maximum.projected_logits
        && work.attention_pairs <= maximum.attention_pairs && work.projections.dot_products <= maximum.projections.dot_products
        && work.projections.multiply_accumulates <= maximum.projections.multiply_accumulates
}

#[cfg(test)] mod tests;
