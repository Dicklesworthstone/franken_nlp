//! Complete original documents on one owned native extraction stream.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        extract::grouped_int4::long::{cohort::Int4ExtractionMapCohortBudget,
            stream::{Int4ExtractionDocumentBatchPlanner, NativeInt4DocumentExtractionBatch,
                Int4ExtractionDocumentAdmission, Int4ExtractionDocumentBatchAdmission,
                memory::{Int4ExtractionDocumentMemoryLimits, Int4ExtractionDocumentMemoryRequirement}}}},
    hosted::corpus::CorpusLimits,
    grammar::CompileLimits,
    native_engine::{constrained_int4::sparse::Int4JsonSparseLimits, lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::extract::object_reconcile::RootObjectLimits,
    tokenizer::untrusted::SafeBpeLimits,
};

struct StreamInput<R, W> {
    preparation: Option<(Int4ExtractionDocumentBatchPlanner, Arc<ExtractionVocabulary>)>,
    reader: R,
    writer: W,
}

/// Quote complete transport, native, reduction, identity and result ownership
/// before model/population IO. The pure core helper also serves CLI builds
/// without a hosted runtime. Compiler/schema/vocabulary and supplied IO owners
/// must fit the caller's explicit preparation and IO declarations.
pub fn int4_extraction_document_corpus_requirements(width: usize, budget: TaskBudget,
    document: Int4ExtractionMapLimits, limits: CorpusLimits, reduction_reserve_bytes: u64,
    cohort_chunks: Option<usize>) -> Result<Int4ExtractionDocumentMemoryRequirement, HostedError> {
    limits.native.run.validate()?;
    memory_limits(limits, reduction_reserve_bytes, cohort_chunks)
        .requirements(width, budget, document, limits.transport).map_err(map_error)
}

/// Complete stream ownership with one optional bounded input-encoding
/// workspace in addition to the unchanged retained preparation declaration.
#[allow(clippy::too_many_arguments)]
pub fn int4_extraction_document_corpus_requirements_with_encoding(width: usize, budget: TaskBudget,
    document: Int4ExtractionMapLimits, limits: CorpusLimits, reduction_reserve_bytes: u64,
    cohort_chunks: Option<usize>, encoding: &UntrustedEncoding)
    -> Result<Int4ExtractionDocumentMemoryRequirement, HostedError> {
    limits.native.run.validate()?;
    memory_limits(limits, reduction_reserve_bytes, cohort_chunks)
        .requirements_with_encoding(width, budget, document, limits.transport, encoding).map_err(map_error)
}

/// The same complete stream quote with an explicit original-schema root-object
/// reducer. Both schema parses and the complete reconciliation workspace coexist
/// with the retained chunk map, before any original source is read or prepared.
#[allow(clippy::too_many_arguments)]
pub fn int4_extraction_document_object_corpus_requirements(width: usize, budget: TaskBudget,
    document: Int4ExtractionMapLimits, limits: CorpusLimits, reduction_reserve_bytes: u64,
    cohort_chunks: Option<usize>, compiler: CompileLimits, object: Option<RootObjectLimits>)
    -> Result<Int4ExtractionDocumentMemoryRequirement, HostedError> {
    limits.native.run.validate()?;
    memory_limits(limits, reduction_reserve_bytes, cohort_chunks)
        .requirements_with_root_object(width, budget, document, limits.transport, compiler, object)
        .map_err(map_error)
}

/// The common pre-IO quote for the fixed document semantics and input policy.
/// Compact exact-schema maps and root arrays are supported; compact partial
/// root-object reconciliation is refused by both this quote and its compiler.
#[allow(clippy::too_many_arguments)]
pub fn int4_extraction_document_object_corpus_requirements_with_encoding(width: usize, budget: TaskBudget,
    document: Int4ExtractionMapLimits, limits: CorpusLimits, reduction_reserve_bytes: u64,
    cohort_chunks: Option<usize>, compiler: CompileLimits, object: Option<RootObjectLimits>,
    encoding: &UntrustedEncoding) -> Result<Int4ExtractionDocumentMemoryRequirement, HostedError> {
    limits.native.run.validate()?;
    memory_limits(limits, reduction_reserve_bytes, cohort_chunks)
        .requirements_with_root_object_and_encoding(width, budget, document, limits.transport, compiler, object, encoding)
        .map_err(map_error)
}

fn memory_limits(limits: CorpusLimits, reduction_reserve_bytes: u64, cohort_chunks: Option<usize>)
    -> Int4ExtractionDocumentMemoryLimits {
    Int4ExtractionDocumentMemoryLimits { context_tokens: limits.native.context_tokens,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_reserve_bytes, reduction_reserve_bytes, cohort_chunks }
}

impl NlpEngine {
    /// Stream complete original documents using a fixed exact schema and one
    /// resident model. Every chunk retains its independent exact JSON value
    /// and original coordinates. An explicitly configured root-object planner
    /// additionally reconciles complementary properties against the untouched
    /// original schema; conflicts or missing required fields refuse the document.
    /// Native work, masks and all verification axes share one nonrenewable
    /// invocation ledger. A failed chunk publishes no partial document result.
    ///
    /// Readers, writers, compiler, vocabulary and native state remain owned
    /// through physical drain. Output guards survive serialization/delivery.
    /// Flush changes only the duplicate-ID epoch; it grants no new resources.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionDocumentBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        limits: CorpusLimits, reduction_reserve_bytes: u64, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        check_identity(planner.base_execution_identity(), planner.group_width())?;
        if planner.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        if planner.retained_defaults_bytes().map_err(HostedError::BatchSetup)? > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("extraction defaults exceed their preparation reservation"));
        }
        let document = planner.document_limits();
        let cohort_chunks = planner.uses_chunk_cohorts().then_some(document.mapping.reduction.map_batch_chunks);
        let required = int4_extraction_document_object_corpus_requirements_with_encoding(planner.group_width(),
            planner.document_budget(), document, limits, reduction_reserve_bytes, cohort_chunks,
            planner.compiler_limits(), planner.root_object_reconciliation(), planner.document_encoding())?;
        let cohort = cohort_chunks.map(|_| planner.cohort_requirements().map_err(HostedError::BatchSetup)).transpose()?;
        let capacity = DocumentCapacity { document, budget: planner.document_budget(), selected_rows: planner.selected_rows(),
            encoding: match planner.document_encoding() {
                UntrustedEncoding::ByteFallback => None, UntrustedEncoding::SafeBpeV1(limits) => Some(*limits),
            },
            kv_bytes: required.native.kv_bytes, max_result_bytes: required.max_output_bytes };
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { preparation: Some((planner, vocabulary)), reader, writer }))?;
        let frontier = Pending::reserve(&lease, MemoryClass::JobBuffers, required.document_workspace_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            poll(control)?;
            let (planner, vocabulary) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let admission = DocumentAdmission { lease: &lease, model: model.model_identity(),
                base: planner.base_execution_identity(), capacity };
            let result = if let Some(quote) = cohort {
                let budget = Int4ExtractionMapCohortBudget { context_limit: limits.native.context_tokens,
                    native_memory: Int4MemoryBudget { max_kv_bytes: quote.native.kv_bytes,
                        max_rope_bytes: quote.native.rope_bytes,
                        max_scratch_payload_bytes: quote.native.scratch_payload_bound },
                    max_workspace_bytes: quote.workspace_bytes, max_result_bytes: quote.max_result_bytes };
                // Peak reservations remain live while exact per-window engines
                // allocate, execute and drain. No unused scalar engine coexists.
                let result = {
                    let mut processor = NativeInt4DocumentExtractionBatch::new_cohorts(&planner,
                        &model.inner.loaded.value, &vocabulary, budget, kernel, admission)
                        .map_err(HostedError::BatchSetup)?;
                    stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)
                };
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
                    let mut processor = NativeInt4DocumentExtractionBatch::new(&planner, &mut engine.value,
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
struct DocumentCapacity {
    document: Int4ExtractionMapLimits,
    budget: TaskBudget,
    selected_rows: Option<Int4JsonSparseLimits>,
    encoding: Option<SafeBpeLimits>,
    kv_bytes: u64,
    max_result_bytes: u64,
}
struct DocumentAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    base: &'a ExecutionIdentity,
    capacity: DocumentCapacity,
}
impl Int4ExtractionDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4ExtractionDocumentAdmission<'_>)
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
        let guard = output_claim(self.lease, request.max_result_bytes, request.max_output_tokens)
            .map_err(|error| match error { HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
                _ => BatchItemFailure::fatal(BatchCode::InvalidExecution) })?;
        Ok((identities, guard))
    }
}

fn check_request(width: usize, base: &ExecutionIdentity, capacity: DocumentCapacity,
    request: &Int4ExtractionDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    let c = capacity; let m = c.document.mapping;
    if base.task_spec != "extract-v1" || request.chunk_count == 0 || request.chunk_count > m.chunks.max_chunks
        || request.chunk_count > 256 || request.identities.len() != request.chunk_count
        || request.document_encoding != c.encoding.map_or(UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1)
        || request.kv_reservation_bytes != c.kv_bytes || c.kv_bytes == 0 {
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
    if m.mask_visits_per_chunk.checked_mul(request.chunk_count as u64) != Some(request.mask_node_visits)
        || request.mask_node_visits == 0 || request.mask_node_visits > m.max_mask_visits
        || request.mask_limits.max_trie_node_visits != m.mask_limits.max_trie_node_visits
        || request.mask_limits.checkpoint_interval_nodes != m.mask_limits.checkpoint_interval_nodes {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let rows_per_token = match c.selected_rows {
        Some(limits) => limits.validate().map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))? as u64,
        None => NANBEIGE_VOCAB_SIZE as u64,
    };
    // Admission receives planned ceilings, including terminal EOS for every
    // chunk. Bind the whole allowance to the factory's exact projection policy;
    // actual sparse rows and a divisible subtotal cannot renew or reduce it.
    let tokens = u64::from(c.budget.max_output_tokens).checked_mul(request.chunk_count as u64)
        .filter(|&tokens| tokens != 0).ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    let projected = tokens.checked_mul(rows_per_token)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    let work = request.model_work;
    if [work.forward_positions, work.projected_logits, work.attention_pairs,
        work.projections.dot_products, work.projections.multiply_accumulates].contains(&0)
        || work.projected_logits != projected || tokens > work.forward_positions
        || !work_fits(work, m.max_model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if c.max_result_bytes == 0 || c.max_result_bytes != m.reduction.max_result_bytes as u64
        || request.max_result_bytes != c.max_result_bytes
        || tokens != request.max_output_tokens {
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
