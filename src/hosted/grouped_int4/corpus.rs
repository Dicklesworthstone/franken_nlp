//! Ordered NDJSON extraction on one charged resident grouped-INT4 engine.
//! Model, vocabulary, transport and native workspace live for the entire
//! blocking invocation. Corpus and mask budgets never renew at an epoch.

use super::*;
mod source;
pub use source::Int4SourceCorpusConfig;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        extract::{ExtractionMaskBudget, grouped_int4::{Int4ExtractionAdmission,
            Int4ExtractionBatchAdmission, Int4ExtractionBatchLimits,
            Int4ExtractionBatchPlanner, NativeInt4ExtractionBatch}}},
    hosted::corpus::CorpusLimits,
    native_engine::{constrained_int4, kv::KV_BYTES_PER_TOKEN,
        lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
};

struct StreamInput<R, W> {
    preparation: Option<(Int4ExtractionBatchPlanner, Arc<ExtractionVocabulary>)>,
    reader: R,
    writer: W,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Requirements {
    native: Int4MemoryRequirement,
    workspace_bytes: u64,
    input_bytes: u64,
}

impl NlpEngine {
    /// Run raw-schema/source extraction as the existing ordered NDJSON
    /// protocol. One native engine and vocabulary serve the complete stream;
    /// each document uses its own sealed identity and logical KV session.
    ///
    /// Preparation reservations must price the owned planner, vocabulary and
    /// compiler work; IO reservations must price the supplied reader/writer.
    /// Both owners move into the blocking invocation and outlive all delivery.
    /// Kernel selection is explicit. Blocking IO remains synchronous, with
    /// cancellation waiting for physical cleanup before this method returns.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        if planner.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let required = requirements(model.model_identity().group_width(), native_work, limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { preparation: Some((planner, vocabulary)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let (planner, vocabulary) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, workspace, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let admission = CorpusAdmission { lease: &lease, model: model.model_identity(),
                kv_bytes: required.native.kv_bytes, masks: native_work.masks,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let mut processor = NativeInt4ExtractionBatch::new(
                    planner, &mut engine.value, &vocabulary, admission, native_work,
                ).map_err(HostedError::BatchSetup)?;
                stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(vocabulary);
            drop(input); // readers/writers drop before their charge and physical completion
            drop(lease);
            result
        })
    }
}

fn requirements(width: usize, work: Int4ExtractionBatchLimits, limits: CorpusLimits)
    -> Result<Requirements, HostedError> {
    work.validate().map_err(HostedError::BatchSetup)?;
    let input_bytes = limits.reservation_bytes()?;
    if limits.native.allocator_reserve_bytes == 0 {
        return Err(HostedError::Limits("INT4 corpus allocator reserve"));
    }
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    Ok(Requirements { native, workspace_bytes, input_bytes })
}

struct CorpusAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    kv_bytes: u64,
    masks: ExtractionMaskBudget,
    output_bytes: u64,
}
impl Int4ExtractionBatchAdmission for CorpusAdmission<'_> {
    type Guard = Pending;

    fn admit(&mut self, request: Int4ExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        self.admit_for_task(request, "extract-v1")
    }
}
impl CorpusAdmission<'_> {
    fn admit_for_task(&self, request: Int4ExtractionAdmission<'_>, task_spec: &str)
        -> Result<(ExecutionIdentity, Pending), BatchItemFailure> {
        constrained_int4::check_model(request.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_request_for_task(self.model.group_width(), self.kv_bytes, self.output_bytes,
            self.masks, &request, task_spec)?;
        let tokens = request.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64;
        // Keep the reservation live through result allocation, serialization
        // and writer delivery. Allocation has not happened at admission, so
        // no committed-memory receipt is fabricated here. GuardedOutput drops
        // its owned result before Pending aborts this reservation.
        let guard = output_claim(self.lease, request.max_result_bytes, tokens).map_err(admission_failure)?;
        Ok((request.identity.clone(), guard))
    }
}

#[cfg(test)]
fn check_request(width: usize, kv_bytes: u64, output_bytes: u64,
    masks: ExtractionMaskBudget, request: &Int4ExtractionAdmission<'_>) -> Result<(), BatchItemFailure> {
    check_request_for_task(width, kv_bytes, output_bytes, masks, request, "extract-v1")
}

fn check_request_for_task(width: usize, kv_bytes: u64, output_bytes: u64,
    masks: ExtractionMaskBudget, request: &Int4ExtractionAdmission<'_>, task_spec: &str)
    -> Result<(), BatchItemFailure> {
    constrained_int4::check_profile(request.identity, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if !matches!(task_spec, "extract-v1" | "ner-v1" | "keyphrases-v1" | "summarize-v1" | "answer-v1")
        || request.identity.task_spec != task_spec || request.kv_reservation_bytes != kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    if request.mask_node_visits == 0 || request.mask_node_visits != masks.max_visits_per_item
        || request.mask_limits.max_trie_node_visits != masks.per_mask.max_trie_node_visits
        || request.mask_limits.checkpoint_interval_nodes != masks.per_mask.checkpoint_interval_nodes {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let work = request.model_work;
    if work.forward_positions == 0 || work.projected_logits == 0
        || work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || work.projected_logits / NANBEIGE_VOCAB_SIZE as u64 > work.forward_positions {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if work.forward_positions > kv_bytes / KV_BYTES_PER_TOKEN as u64 {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let positions = usize::try_from(work.forward_positions)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    let rows = usize::try_from(work.projected_logits)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if Int4Work::for_sequence(width, 0, positions, rows).ok() != Some(work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if request.max_result_bytes == 0 || request.max_result_bytes > output_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}

fn admission_failure(error: HostedError) -> BatchItemFailure {
    match error {
        HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}

#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod jobs;

#[cfg(test)]
mod tests;
