//! A bounded resident stream of complete sentiment documents.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        int4_document::{fits, nonzero},
        sentiment::grouped_int4::{SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES,
            Int4SentimentDocumentAdmission, Int4SentimentDocumentBatchAdmission,
            Int4SentimentDocumentBatchPlanner, NativeInt4SentimentDocumentBatch}},
    hosted::corpus::CorpusLimits,
    native_engine::{constrained_int4, strict_int4::cohort::MAX_INT4_SEQUENCES},
    tasks::sentiment::{SentimentLimits, grouped_int4::long::{Int4SentimentMapError,
        Int4SentimentMapLimits, memory::{Int4SentimentMapMemoryLimits, Int4SentimentMapMemoryRequirement}}},
};

/// Pure checked reservation before model or source IO. The complete-map quote
/// owns one plan set and all original chunk evidence; the stream additionally
/// owns transport staging, original/frozen identities, defaults and supplied IO.
/// `max_axes` limits every override, independently of the selected defaults.
/// These are explicit reservations, not observed process RSS measurements.
#[allow(clippy::too_many_arguments)]
pub fn int4_sentiment_document_corpus_requirements(width: usize, task: TaskBudget,
    planning: SentimentLimits, mapping: Int4SentimentMapLimits, limits: CorpusLimits,
    max_axes: usize, cohort_chunks: Option<usize>) -> Result<Int4SentimentMapMemoryRequirement, HostedError> {
    let transport = limits.reservation_bytes()?;
    if !(1..=4).contains(&max_axes) || cohort_chunks.is_some_and(|chunks| chunks == 0
        || chunks > mapping.chunks.max_chunks || chunks.checked_mul(max_axes).is_none_or(|n| n > MAX_INT4_SEQUENCES)) {
        return Err(HostedError::Limits("sentiment document cohort geometry"));
    }
    let max_result_bytes = document_output_bytes(mapping, cohort_chunks)?;
    if max_result_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("complete sentiment document output reservation"));
    }
    let mut required = map_memory(limits, max_axes, cohort_chunks)
        .requirements(width, task, planning, mapping).map_err(map_error)?;
    let metadata = (mapping.chunks.max_chunks as u64).checked_mul(SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("sentiment document metadata arithmetic"))?;
    required.input_bytes = sum(&[required.input_bytes, transport, metadata])?;
    sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
        required.output_reservation_bytes])?;
    Ok(required)
}
fn map_memory(limits: CorpusLimits, max_axes: usize, cohort_chunks: Option<usize>) -> Int4SentimentMapMemoryLimits {
    Int4SentimentMapMemoryLimits { context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes, max_axes, cohort_chunks }
}
pub(super) fn document_output_bytes(mapping: Int4SentimentMapLimits, cohort_chunks: Option<usize>) -> Result<u64, HostedError> {
    mapping.max_result_bytes.checked_add(if cohort_chunks.is_some() { 4096 } else { 0 })
        .ok_or(HostedError::Limits("sentiment document output arithmetic"))
}

struct StreamInput<R, W> { planner: Option<Int4SentimentDocumentBatchPlanner>, reader: R, writer: W }

struct DeferredSentimentStream<R, W> {
    completion: stream::deferred::DeferredCompletion,
    input: Charged<StreamInput<R, W>>,
}

impl NlpEngine {
    /// Stream complete original sources under one resident model, deadline and
    /// five-axis work ledger. Every axis and chunk completes before one result
    /// is published; flush never replenishes model work. Serial execution
    /// reuses one cache; explicit cohorts use real simultaneous chunk heads.
    ///
    /// Supplied readers/writers, compiler/defaults and prepared sources remain
    /// in the owned physical invocation. IO remains cooperative; this method
    /// does not create a runtime, detach a worker or reload a model per item.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sentiment_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4SentimentDocumentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sentiment_documents(model, planner, limits, kernel, None,
            reader, writer, cancellation)
    }

    /// Keep one persistent numerical row team across every axis, chunk and
    /// original document. The same planner, policy and complete-map finalizer
    /// retain their nonrenewable native and compact-tokenizer allowances.
    /// Terminal completion is published after physical worker join under the
    /// original run deadline. Native chunk cohorts cannot be combined.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_sentiment_documents_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: Int4SentimentDocumentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_sentiment_documents(model, planner, limits, kernel, Some(compute_threads),
            reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_sentiment_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4SentimentDocumentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: Option<usize>, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width()
            || planner.base_execution_identity().task_spec != "sentiment-v1" {
            return Err(HostedError::ModelIdentity);
        }
        let mapping = planner.mapping_limits();
        crate::batch::sentiment::grouped_int4::validate_limits(planner.task_budget(), planner.planning_limits(),
            mapping, planner.max_run_work()).map_err(HostedError::BatchSetup)?;
        let max_axes = planner.max_axes(); let cohort_chunks = planner.cohort_chunks();
        check_team_schedule(compute_threads, cohort_chunks)?;
        let required = int4_sentiment_document_corpus_requirements(planner.group_width(), planner.task_budget(),
            planner.planning_limits(), mapping, limits, max_axes, cohort_chunks)?
            .with_input_encoding(planner.document_encoding()).map_err(map_error)?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, planner.group_width())).transpose()?.unwrap_or(0);
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
            required.output_reservation_bytes, team_scratch])?;
        if planner.retained_defaults_bytes().map_err(HostedError::BatchSetup)? > limits.preparation_reserve_bytes {
            return Err(HostedError::Limits("sentiment defaults exceed their preparation reservation"));
        }
        let capacity = DocumentCapacity { context: limits.native.context_tokens, kv_bytes: required.native.kv_bytes,
            mapping, max_axes, cohort_chunks, max_result_bytes: planner.max_result_bytes() };
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { planner: Some(planner), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let model = model.clone();
        match compute_threads {
        None => dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let planner = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let admission = DocumentAdmission { lease: &lease, model: model.model_identity(),
                capacity, output_bytes: required.output_reservation_bytes };
            let result = if cohort_chunks.is_some() {
                // Retain the entire declared active peak while exact ragged
                // window engines are allocated, used, cleared and destroyed.
                let kv = kv.commit()?; let scratch = scratch.commit()?;
                let result = {
                    let mut processor = NativeInt4SentimentDocumentBatch::new_cohorts(planner,
                        &model.inner.loaded.value, kernel, map_memory(limits, max_axes, cohort_chunks), admission)
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
                    let mut processor = NativeInt4SentimentDocumentBatch::new(planner, &mut engine.value, admission)
                        .map_err(HostedError::BatchSetup)?;
                    stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)
                };
                drop(engine);
                result
            };
            drop(input); // IO and retained compiler storage precede physical completion.
            drop(lease);
            result
        }),
        Some(width) => dispatch::run_int4_team_with_finalize(self, limits.native.run, cancellation,
            model, width, kernel, move |weights, control, team| {
                let mut input = input;
                checkpoint(control)?;
                let planner = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
                let admission = DocumentAdmission { lease: &lease, model: weights.identity(),
                    capacity, output_bytes: required.output_reservation_bytes };
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(weights, limits.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound,
                    }).map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let completion = {
                    let mut processor = NativeInt4SentimentDocumentBatch::new(planner,
                        &mut engine.value, admission).map_err(HostedError::BatchSetup)?;
                    stream::deferred::run_ndjson_deferred(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)?
                };
                drop(engine); drop(lease);
                Ok(DeferredSentimentStream { completion, input })
            }, |pending, control| pending.publish(control)),
        }
    }
}

fn check_team_schedule(threads: Option<usize>, cohort_chunks: Option<usize>) -> Result<(), HostedError> {
    if threads.is_some() && cohort_chunks.is_some() {
        return Err(HostedError::Limits("INT4 sentiment corpus team cannot combine with native chunk cohorts"));
    }
    Ok(())
}

impl<R, W: Write> DeferredSentimentStream<R, W> {
    fn publish<C: crate::native_engine::decode::DecodeStepControl>(self, control: &mut C)
        -> Result<BatchSummary, HostedError> {
        let Self { completion, mut input } = self;
        let result = completion.publish(&mut input.value.writer, control).map_err(HostedError::Batch);
        drop(input);
        result
    }
}

#[derive(Clone, Copy)]
pub(super) struct DocumentCapacity {
    pub(super) context: usize,
    pub(super) kv_bytes: u64,
    pub(super) mapping: Int4SentimentMapLimits,
    pub(super) max_axes: usize,
    pub(super) cohort_chunks: Option<usize>,
    pub(super) max_result_bytes: u64,
}
pub(super) struct DocumentAdmission<'a> {
    pub(super) lease: &'a EngineLease,
    pub(super) model: &'a Int4ModelIdentity,
    pub(super) capacity: DocumentCapacity,
    pub(super) output_bytes: u64,
}
impl Int4SentimentDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4SentimentDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Pending), BatchItemFailure> {
        check_request(self.model.group_width(), self.capacity, &request)?;
        for identity in request.identities {
            constrained_int4::check_model(identity, self.model)
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        let mut identities = Vec::new();
        identities.try_reserve_exact(request.identities.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        identities.extend(request.identities.iter().cloned());
        let guard = Pending::reserve(self.lease, MemoryClass::JobBuffers, self.output_bytes)
            .map_err(|e| match e { HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
                _ => BatchItemFailure::fatal(BatchCode::InvalidExecution) })?;
        Ok((identities, guard))
    }
}
fn check_request(width: usize, capacity: DocumentCapacity, request: &Int4SentimentDocumentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let c = capacity;
    if request.group_width != width || request.kv_reservation_bytes != c.kv_bytes
        || request.task_budget.validate().is_err() || request.identities.is_empty()
        || request.identities.len() > c.mapping.chunks.max_chunks
        || request.required_context == 0 || request.required_context > c.context
        || !(1..=4).contains(&c.max_axes) || request.head_count == 0 || request.head_count > c.max_axes
        || request.cohort_chunks != c.cohort_chunks {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let row = Int4MemoryRequirement::for_context(c.context, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let physical = if let Some(chunks) = c.cohort_chunks {
        let heads = chunks.checked_mul(c.max_axes).filter(|heads| (1..=MAX_INT4_SEQUENCES).contains(heads))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
        if chunks > c.mapping.chunks.max_chunks { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let contexts = [c.context; MAX_INT4_SEQUENCES];
        Int4MemoryRequirement::for_cohort(&contexts[..heads], width)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?
    } else { row };
    if physical.kv_bytes != c.kv_bytes || row.kv_bytes.checked_mul(request.head_count as u64)
        .is_none_or(|bytes| bytes > request.task_budget.max_kv_bytes) {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    if !nonzero(request.model_work) || !fits(request.model_work, c.mapping.max_model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let expected_output = document_output_bytes(c.mapping, c.cohort_chunks)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::OutputLineLimit))?;
    if expected_output == 0 || c.max_result_bytes != expected_output || request.max_result_bytes != expected_output {
        return Err(BatchItemFailure::fatal(BatchCode::OutputLineLimit));
    }
    for identity in request.identities {
        identity.validate().map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        constrained_int4::check_profile(identity, width).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if identity.task_spec != "sentiment-v1" { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    }
    Ok(())
}
fn map_error(error: Int4SentimentMapError) -> HostedError { HostedError::Int4(Int4HostError::SentimentMap(error)) }

#[cfg(test)] mod tests;
