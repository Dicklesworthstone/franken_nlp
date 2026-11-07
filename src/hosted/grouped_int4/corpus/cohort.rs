//! Bounded extraction windows on the real multi-sequence INT4 decoder.
use super::*;
use crate::native_engine::strict_int4::prefill::Int4PrefillLimits;
use crate::batch::{grouped::{self, GroupLimits}, extract::grouped_int4::{
    Int4ExtractionCohortBatchLimits, Int4ExtractionCohortBatchStats, NativeInt4ExtractionCohortBatch,
    memory::Int4ExtractionCohortMemoryLimits,
}};

/// Complete semantic results remain in their existing ordered NDJSON records.
/// Physical decoder statistics are returned once for the entire invocation.
pub struct Int4ExtractionCohortBatchResult {
    pub summary: BatchSummary,
    pub cohorts: Int4ExtractionCohortBatchStats,
}

impl NlpEngine {
    /// Run independent schema/source documents through shared decoder layers.
    /// Each document retains its grammar, original bytes, independent KV and
    /// sealed greedy projection policy. No source mask crosses a document.
    ///
    /// Preparation is priced for the full window plus one lookahead. Native
    /// maximum payload reservations precede IO; each exact-size engine lives
    /// within them. All result guards stay together through the final flush.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract_cohorts<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, max_documents: usize,
        kernel: PackedInt4Kernel, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<Int4ExtractionCohortBatchResult, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.batch_int4_extract_cohorts_for_schedule(model, planner, vocabulary, native_work, limits,
            max_documents, kernel, None, reader, writer, cancellation)
    }

    /// Run the same ordered corpus with bounded packed prompt/feedback passes.
    /// The total-token cap is independent of the document window. Extra native
    /// scratch is reserved before IO and retained until physical completion.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract_cohorts_packed<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, max_documents: usize,
        kernel: PackedInt4Kernel, prefill: Int4PrefillLimits,
        reader: R, writer: W, cancellation: CancellationToken)
        -> Result<Int4ExtractionCohortBatchResult, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.batch_int4_extract_cohorts_for_schedule(model, planner, vocabulary, native_work, limits,
            max_documents, kernel, Some(prefill), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_int4_extract_cohorts_for_schedule<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, max_documents: usize,
        kernel: PackedInt4Kernel, prefill: Option<Int4PrefillLimits>,
        reader: R, writer: W, cancellation: CancellationToken)
        -> Result<Int4ExtractionCohortBatchResult, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        if planner.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let selected_rows = planner.selected_rows();
        let memory = Int4ExtractionCohortMemoryLimits {
            max_documents, context_tokens: limits.native.context_tokens,
            preparation_reserve_bytes: limits.preparation_reserve_bytes,
            io_reserve_bytes: limits.io_reserve_bytes,
            allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
        };
        let required = memory.requirements_for_execution(planner.group_width(), native_work,
            limits.transport, selected_rows, prefill).map_err(HostedError::BatchSetup)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { preparation: Some((planner, vocabulary)), reader, writer }))?;
        // Keep these as live reservations across changing exact-size windows;
        // an empty input does not fabricate an allocated native payload.
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let native = Int4ExtractionCohortBatchLimits {
            max_documents, context_limit: limits.native.context_tokens,
            native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound },
            max_workspace_bytes: required.decode_workspace_bytes,
            max_item_result_bytes: limits.transport.max_output_line_bytes as u64,
            max_result_bytes: required.max_result_bytes,
        };
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let native_guards = (kv, workspace);
            checkpoint(control)?;
            let (planner, vocabulary) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let admission = CohortAdmission { lease: &lease, model: model.model_identity(),
                context_limit: limits.native.context_tokens, masks: native_work.masks,
                output_bytes: limits.transport.max_output_line_bytes as u64, selected_rows };
            let mut processor = match prefill {
                Some(prefill) => NativeInt4ExtractionCohortBatch::new_packed(planner, &model.inner.loaded.value,
                    &vocabulary, admission, native_work, native, kernel, prefill),
                None => NativeInt4ExtractionCohortBatch::new(planner, &model.inner.loaded.value,
                    &vocabulary, admission, native_work, native, kernel),
            }.map_err(HostedError::BatchSetup)?;
            let result = grouped::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                &mut processor, limits.transport, GroupLimits { max_records: max_documents }, control)
                .map_err(HostedError::Batch);
            let cohorts = processor.cohort_stats();
            drop(processor); drop(vocabulary); drop(input); drop(native_guards); drop(lease);
            result.map(|summary| Int4ExtractionCohortBatchResult { summary, cohorts })
        })
    }
}

struct CohortAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    context_limit: usize,
    masks: ExtractionMaskBudget,
    output_bytes: u64,
    selected_rows: Option<Int4JsonSparseLimits>,
}
impl Int4ExtractionBatchAdmission for CohortAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4ExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        constrained_int4::check_model(request.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let tokens = check_cohort_request(self.model.group_width(), self.context_limit,
            self.output_bytes, self.masks, &request, self.selected_rows)?;
        let guard = output_claim(self.lease, request.max_result_bytes, tokens).map_err(admission_failure)?;
        Ok((request.identity.clone(), guard))
    }
}
fn check_cohort_request(width: usize, context_limit: usize, output_bytes: u64,
    masks: ExtractionMaskBudget, request: &Int4ExtractionAdmission<'_>,
    selected_rows: Option<Int4JsonSparseLimits>) -> Result<u64, BatchItemFailure> {
    // This route creates the exact preflight context for each independent row;
    // never accept a serial reusable cache as the charge for the whole window.
    let exact_kv = request.model_work.forward_positions.checked_mul(KV_BYTES_PER_TOKEN as u64)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
    if request.model_work.forward_positions > context_limit as u64 || request.kv_reservation_bytes != exact_kv {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    check_request_for_projection(width, exact_kv, output_bytes, masks, request, "extract-v1", selected_rows)
}

#[cfg(test)] mod tests;
