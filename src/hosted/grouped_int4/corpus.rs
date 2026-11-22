//! Ordered NDJSON extraction on one charged resident grouped-INT4 engine.
//! Model, vocabulary, transport and native workspace live for the entire
//! blocking invocation. Corpus and mask budgets never renew at an epoch.

use super::*;
mod source;
pub use source::{Int4SourceCorpusConfig, Int4SourceCohortBatchResult};
mod cohort;
pub use cohort::Int4ExtractionCohortBatchResult;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        extract::{ExtractionMaskBudget, grouped_int4::{Int4ExtractionAdmission,
            Int4ExtractionBatchAdmission, Int4ExtractionBatchLimits,
            Int4ExtractionBatchPlanner, NativeInt4ExtractionBatch, memory::encoding_workspace_bytes}}},
    hosted::corpus::CorpusLimits,
    native_engine::{constrained_int4::{self, sparse::Int4JsonSparseLimits}, kv::KV_BYTES_PER_TOKEN,
        lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::{Int4Work, prefill::Int4PrefillLimits}},
    tokenizer::untrusted::UntrustedEncoding,
};

struct StreamInput<R, W> {
    preparation: Option<(Int4ExtractionBatchPlanner, Arc<ExtractionVocabulary>)>,
    reader: R,
    writer: W,
}

/// Both structured stream hosts retain the complete charged IO aggregate
/// across worker join. The staged footer drops before that aggregate's charge
/// on cancellation or unwind; successful publication consumes it exactly once.
struct DeferredStream<I> {
    completion: Result<stream::deferred::DeferredCompletion, HostedError>,
    input: Charged<I>,
    native: Option<(CommittedMemory, CommittedMemory)>,
    lease: EngineLease,
}
impl<I> DeferredStream<I> {
    fn publish<W: Write>(self, control: &mut RunControl, writer: fn(&mut I) -> &mut W)
        -> Result<BatchSummary, HostedError> {
        let Self { completion, mut input, native, lease } = self;
        // The native engine already dropped its buffers. Its reservations
        // remain live through physical worker join, then release before the
        // terminal writer runs under the original invocation's control.
        drop(native);
        let result = completion.and_then(|completion|
            completion.publish(writer(&mut input.value), control).map_err(HostedError::Batch));
        drop(input);
        drop(lease);
        result
    }
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
    /// The planner's fixed compact-encoding workspace is quoted separately and
    /// automatically retained with the input aggregate for this whole stream.
    /// Both owners move into the blocking invocation and outlive all delivery.
    /// Kernel selection is explicit. Blocking IO remains synchronous, with
    /// cancellation waiting for physical cleanup before this method returns.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_extract_batch(model, planner, vocabulary, native_work, limits, kernel,
            None, None, reader, writer, cancellation)
    }

    /// Execute each independent record with bounded layer-major prompt
    /// morsels. One extra prompt workspace spans the entire ordered stream;
    /// no document cohort or producer-side buffering is required.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract_layer_major<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, kernel: PackedInt4Kernel,
        prefill: Int4PrefillLimits, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_extract_batch(model, planner, vocabulary, native_work, limits, kernel,
            Some(prefill), None, reader, writer, cancellation)
    }

    /// Execute the same extraction stream with one persistent team of 2..=64
    /// compute threads, including the coordinator. All documents and flushes
    /// share the original native-work, mask, deadline and checkpoint ledgers.
    /// Full-vocabulary or explicitly selected legal-row projection keeps its
    /// exact schema/source policy and independent extraction finalizer.
    ///
    /// Worker scratch is admitted in addition to the ordinary stream buffers.
    /// The final run_complete is published inside the same tracked invocation
    /// after every worker joins, with owned IO and its charge still retained.
    /// Blocking writers remain cooperative. This entry uses single-row
    /// prompts; the layer-major variant admits its own bounded row capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_extract_batch(model, planner, vocabulary, native_work, limits, kernel,
            None, Some(compute_threads), reader, writer, cancellation)
    }

    /// Reuse one fixed CPU team across prompt morsels, constrained decoding
    /// and every flush in the stream. Coordinator and worker row buffers are
    /// independently admitted. Complete results retain their schema/source
    /// policy and lifetime work; run_complete is published after worker join.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract_layer_major_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, prefill: Int4PrefillLimits, reader: R, writer: W,
        cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_extract_batch(model, planner, vocabulary, native_work, limits, kernel,
            Some(prefill), Some(compute_threads), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_extract_batch<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, kernel: PackedInt4Kernel,
        prefill: Option<Int4PrefillLimits>, compute_threads: Option<usize>,
        reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        if planner.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let selected_rows = planner.selected_rows();
        let required = requirements_with_document_encoding(model.model_identity().group_width(),
            native_work, limits, planner.document_encoding())?.with_prefill_workspace(prefill)?;
        let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
        required.check_team_workspace(compute_threads, model.model_identity().group_width(), max_batch_rows)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { preparation: Some((planner, vocabulary)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        match compute_threads {
        None => dispatch::run(self, limits.native.run, cancellation, move |control| {
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
                output_bytes: limits.transport.max_output_line_bytes as u64, selected_rows };
            let result = {
                let mut processor = match prefill {
                    None => NativeInt4ExtractionBatch::new_with_document_encoding(
                        planner, &mut engine.value, &vocabulary, admission, native_work),
                    Some(prefill) => NativeInt4ExtractionBatch::new_layer_major(
                        planner, &mut engine.value, &vocabulary, admission, native_work, prefill),
                }.map_err(HostedError::BatchSetup)?;
                stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(vocabulary);
            drop(input); // readers/writers drop before their charge and physical completion
            drop(lease);
            result
        }),
        Some(width) => dispatch::run_int4_prefill_team_with_finalize(self, limits.native.run, cancellation,
            model, width, kernel, max_batch_rows, move |weights, control, team| {
                let mut input = input;
                let mut native = None;
                let completion = (|| {
                    checkpoint(control)?;
                    let (planner, vocabulary) = input.value.preparation.take().ok_or(HostedError::CompletionMissing)?;
                    let mut engine = allocate_native(kv, workspace, || {
                        StrictInt4Engine::new(weights, limits.native.context_tokens, Int4MemoryBudget {
                            max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                            max_scratch_payload_bytes: required.native.scratch_payload_bound,
                        }).map_err(native_error)?.with_kernel(kernel)
                            .with_projection_executor(team).map_err(native_error)
                    })?;
                    let admission = CorpusAdmission { lease: &lease, model: weights.identity(),
                        kv_bytes: required.native.kv_bytes, masks: native_work.masks,
                        output_bytes: limits.transport.max_output_line_bytes as u64, selected_rows };
                    let completion = (|| {
                        let mut processor = match prefill {
                            None => NativeInt4ExtractionBatch::new_with_document_encoding(
                                planner, &mut engine.value, &vocabulary, admission, native_work),
                            Some(prefill) => NativeInt4ExtractionBatch::new_layer_major(
                                planner, &mut engine.value, &vocabulary, admission, native_work, prefill),
                        }.map_err(HostedError::BatchSetup)?;
                        stream::deferred::run_ndjson_deferred(&mut input.value.reader, &mut input.value.writer,
                            &mut processor, limits.transport, control).map_err(HostedError::Batch)
                    })();
                    let ChargedNative { value, _kv, _scratch } = engine;
                    drop(value);
                    native = Some((_kv, _scratch));
                    drop(vocabulary);
                    completion
                })();
                Ok(DeferredStream { completion, input, native, lease })
            }, |pending, control| pending.publish(control, |input| &mut input.writer)),
        }
    }
}

impl Requirements {
    fn with_prefill_workspace(mut self, prefill: Option<Int4PrefillLimits>) -> Result<Self, HostedError> {
        self.workspace_bytes = super::prefill::with_extra_workspace(self.workspace_bytes, prefill)?;
        sum(&[self.input_bytes, self.native.kv_bytes, self.workspace_bytes])?;
        Ok(self)
    }
    /// The dispatcher separately reserves this exact worker quote. Check its
    /// addition before allocating any stream/native storage; the serial quote
    /// and all semantic task budgets remain unchanged.
    fn check_team_workspace(self, compute_threads: Option<usize>, group_width: usize, max_batch_rows: usize)
        -> Result<(), HostedError> {
        let extra = compute_threads.map(|threads|
            dispatch::int4_prefill_team_workspace_bytes(threads, group_width, max_batch_rows))
            .transpose()?.unwrap_or(0);
        sum(&[self.input_bytes, self.native.kv_bytes, self.workspace_bytes, extra])?;
        Ok(())
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
fn requirements_with_document_encoding(width: usize, work: Int4ExtractionBatchLimits, limits: CorpusLimits,
    encoding: &UntrustedEncoding) -> Result<Requirements, HostedError> {
    let mut required = requirements(width, work, limits)?;
    required.input_bytes = sum(&[required.input_bytes,
        encoding_workspace_bytes(encoding).map_err(HostedError::BatchSetup)?])?;
    Ok(required)
}

struct CorpusAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    kv_bytes: u64,
    masks: ExtractionMaskBudget,
    output_bytes: u64,
    selected_rows: Option<Int4JsonSparseLimits>,
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
        let tokens = check_request_for_projection(self.model.group_width(), self.kv_bytes, self.output_bytes,
            self.masks, &request, task_spec, self.selected_rows)?;
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

#[cfg(test)]
fn check_request_for_task(width: usize, kv_bytes: u64, output_bytes: u64,
    masks: ExtractionMaskBudget, request: &Int4ExtractionAdmission<'_>, task_spec: &str)
    -> Result<(), BatchItemFailure> {
    check_request_for_projection(width, kv_bytes, output_bytes, masks, request, task_spec, None).map(|_| ())
}

/// The planner fixes this projection policy for the entire serial stream.
/// Planned rows are exactly output-token ceiling times rows per selection;
/// using the vocabulary width for sparse work would underreserve output.
/// Actual sparse row counts never supply admission authority here.
fn check_request_for_projection(width: usize, kv_bytes: u64, output_bytes: u64,
    masks: ExtractionMaskBudget, request: &Int4ExtractionAdmission<'_>, task_spec: &str,
    selected_rows: Option<Int4JsonSparseLimits>) -> Result<u64, BatchItemFailure> {
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
    let rows_per_token = match selected_rows {
        Some(limits) => limits.validate()
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))? as u64,
        None => NANBEIGE_VOCAB_SIZE as u64,
    };
    let work = request.model_work;
    if work.forward_positions == 0 || work.projected_logits == 0
        || work.projected_logits % rows_per_token != 0
        || work.projected_logits / rows_per_token > work.forward_positions {
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
    Ok(work.projected_logits / rows_per_token)
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
