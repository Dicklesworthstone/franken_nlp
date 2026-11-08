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
            Int4ExtractionBatchPlanner, NativeInt4ExtractionBatch}}},
    hosted::corpus::CorpusLimits,
    native_engine::{constrained_int4::{self, sparse::Int4JsonSparseLimits}, kv::KV_BYTES_PER_TOKEN,
        lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
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
    completion: stream::deferred::DeferredCompletion,
    input: Charged<I>,
}
impl<I> DeferredStream<I> {
    fn publish<W: Write>(self, control: &mut RunControl, writer: fn(&mut I) -> &mut W)
        -> Result<BatchSummary, HostedError> {
        let Self { completion, mut input } = self;
        let result = completion.publish(writer(&mut input.value), control).map_err(HostedError::Batch);
        drop(input);
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
            None, reader, writer, cancellation)
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
    /// Blocking writers remain cooperative. No cohort/prefill schedule is
    /// enabled by this entry point, and failed delivery is never retried.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_extract_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_extract_batch(model, planner, vocabulary, native_work, limits, kernel,
            Some(compute_threads), reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_extract_batch<R, W>(&self, model: &ResidentInt4,
        planner: Int4ExtractionBatchPlanner, vocabulary: Arc<ExtractionVocabulary>,
        native_work: Int4ExtractionBatchLimits, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: Option<usize>, reader: R, writer: W, cancellation: CancellationToken)
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
        let required = requirements(model.model_identity().group_width(), native_work, limits)?;
        required.check_team_workspace(compute_threads, model.model_identity().group_width())?;
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
        }),
        Some(width) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
            model, width, kernel, move |weights, control, team| {
                let mut input = input;
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
                let completion = {
                    let mut processor = NativeInt4ExtractionBatch::new(
                        planner, &mut engine.value, &vocabulary, admission, native_work,
                    ).map_err(HostedError::BatchSetup)?;
                    stream::deferred::run_ndjson_deferred(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)?
                };
                drop(engine);
                drop(vocabulary);
                drop(lease);
                Ok(DeferredStream { completion, input })
            }, |pending, control| pending.publish(control, |input| &mut input.writer)),
        }
    }
}

impl Requirements {
    /// The dispatcher separately reserves this exact worker quote. Check its
    /// addition before allocating any stream/native storage; the serial quote
    /// and all semantic task budgets remain unchanged.
    fn check_team_workspace(self, compute_threads: Option<usize>, group_width: usize)
        -> Result<(), HostedError> {
        let extra = compute_threads.map(|threads|
            int4_task_team_workspace_bytes(threads, group_width)).transpose()?.unwrap_or(0);
        sum(&[self.workspace_bytes, extra])?;
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
