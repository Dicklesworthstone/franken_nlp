//! One process-owned invocation for a corpus of complete classification maps.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        classify::grouped_int4::{CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES,
            Int4ClassificationDocumentAdmission, Int4ClassificationDocumentBatchAdmission,
            Int4ClassificationDocumentBatchPlanner, NativeInt4ClassificationDocumentBatch}},
    hosted::corpus::CorpusLimits,
    native_engine::constrained_int4,
    tasks::classify::{ClassificationLimits, grouped_int4::long::{
        Int4ClassificationMapError, Int4ClassificationMapLimits,
        memory::{Int4ClassificationMapMemoryLimits, Int4ClassificationMapMemoryRequirement}}},
};

/// Checked declared memory for the serial complete-document corpus route.
/// Usable before model/input IO. The map owns one set of compiled chunk plans;
/// the stream additionally retains its original request, frozen/admitted
/// identities, exact transport staging and caller-priced compiler/defaults/IO.
/// This is a reservation model, not a measured process RSS bound.
pub fn int4_classification_document_corpus_requirements(width: usize, task: TaskBudget,
    planning: ClassificationLimits, mapping: Int4ClassificationMapLimits, limits: CorpusLimits)
    -> Result<Int4ClassificationMapMemoryRequirement, HostedError> {
    let transport = limits.reservation_bytes()?;
    if mapping.max_result_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("complete classification document output reservation"));
    }
    let mut required = Int4ClassificationMapMemoryLimits { context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes }
        .requirements(width, task, planning, mapping).map_err(map_error)?;
    let metadata = (mapping.chunks.max_chunks as u64).checked_mul(CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("classification document metadata arithmetic"))?;
    required.input_bytes = sum(&[required.input_bytes, transport, metadata])?;
    sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
        required.output_reservation_bytes])?;
    Ok(required)
}

struct StreamInput<R, W> {
    planner: Option<Int4ClassificationDocumentBatchPlanner>, reader: R, writer: W,
}

impl NlpEngine {
    /// Stream bounded original documents larger than one native context using
    /// a single resident model and cache. All chunks finish before one ordered
    /// result is published. Flush never resets the five-axis native ledger.
    ///
    /// Reader, writer, compiler, defaults, compiled plans and output guards
    /// remain inside the existing physical blocking invocation. Arbitrary IO
    /// stays cooperative; this method creates no runtime or detached worker.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_classification_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4ClassificationDocumentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width()
            || planner.base_execution_identity().task_spec != "classify-v1" {
            return Err(HostedError::ModelIdentity);
        }
        let mapping = planner.mapping_limits();
        crate::batch::classify::grouped_int4::validate_limits(planner.task_budget(), planner.planning_limits(),
            mapping, planner.max_run_work()).map_err(HostedError::BatchSetup)?;
        let required = int4_classification_document_corpus_requirements(planner.group_width(), planner.task_budget(),
            planner.planning_limits(), mapping, limits)?;
        if planner.retained_defaults_bytes().map_err(HostedError::BatchSetup)? > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("classification defaults exceed their preparation reservation"));
        }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { planner: Some(planner), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let planner = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = DocumentAdmission { lease: &lease, model: model.model_identity(),
                context: limits.native.context_tokens, kv_bytes: required.native.kv_bytes,
                mapping, output_bytes: required.output_reservation_bytes };
            let result = {
                let mut processor = NativeInt4ClassificationDocumentBatch::new(planner, &mut engine.value, admission)
                    .map_err(HostedError::BatchSetup)?;
                stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(input); // Drop IO owners before their charged storage and physical handoff.
            drop(lease);
            result
        })
    }
}

struct DocumentAdmission<'a> {
    lease: &'a EngineLease, model: &'a Int4ModelIdentity,
    context: usize, kv_bytes: u64, mapping: Int4ClassificationMapLimits, output_bytes: u64,
}
impl Int4ClassificationDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4ClassificationDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Pending), BatchItemFailure> {
        check_request(self.model.group_width(), self.context, self.kv_bytes, self.mapping, &request)?;
        for identity in request.identities {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        identities.extend(request.identities.iter().cloned());
        // The quote covers the complete aggregate, every chunk result and
        // canonical staging. Its pending guard survives sink write AND flush.
        let guard = Pending::reserve(self.lease, MemoryClass::JobBuffers, self.output_bytes)
            .map_err(|e| match e { HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
                _ => BatchItemFailure::fatal(BatchCode::InvalidExecution) })?;
        Ok((identities, guard))
    }
}
fn check_request(width: usize, context: usize, kv: u64, mapping: Int4ClassificationMapLimits,
    request: &Int4ClassificationDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    use crate::batch::classify::grouped_int4::{fits, nonzero};
    if request.group_width != width || request.kv_reservation_bytes != kv
        || request.task_budget.validate().is_err() || request.identities.is_empty()
        || request.identities.len() > mapping.chunks.max_chunks
        || request.required_context == 0 || request.required_context > context {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(context, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if required.kv_bytes != kv || kv > request.task_budget.max_kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    if !nonzero(request.model_work) || !fits(request.model_work, mapping.max_model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    // Aggregate storage is independent of the per-chunk task output budget.
    if request.max_result_bytes == 0 || request.max_result_bytes != mapping.max_result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    for identity in request.identities {
        identity.validate().map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        constrained_int4::check_profile(identity, width).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != "classify-v1" { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    }
    Ok(())
}
fn map_error(error: Int4ClassificationMapError) -> HostedError { HostedError::Int4(Int4HostError::ClassificationMap(error)) }

#[cfg(test)] mod tests;
