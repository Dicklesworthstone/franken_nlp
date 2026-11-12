//! One owned resident stream of complete-source faithfulness judgments.
use super::*;
use std::io::{BufRead, Write};
use crate::{
    batch::{self as stream, BatchCode, BatchItemFailure, BatchSummary,
        int4_document::{fits, nonzero},
        judge::grouped_int4::{FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES,
            Int4FaithfulnessDocumentAdmission, Int4FaithfulnessDocumentBatchAdmission,
            Int4FaithfulnessDocumentBatchPlanner, NativeInt4FaithfulnessDocumentBatch}},
    hosted::corpus::CorpusLimits,
    native_engine::{constrained_int4, strict_int4::cohort::MAX_INT4_SEQUENCES},
    tasks::judge::grouped_int4::faithfulness_map::MAX_FAITHFULNESS_MAP_HEADS,
};

/// Pure reservation before model or source IO. The existing complete-map
/// quote owns one set of plans, the exact claim and source, and every retained
/// witness. The stream also retains transport staging, frozen/admitted chunk
/// identities and caller-supplied IO through physical completion.
///
/// Cohorts price the largest admissible ragged window, capped at 64 native
/// heads. Every actual future window must still fit during preparation;
/// multiplying the requested chunks by the theoretical per-chunk maximum is
/// an allocation bound, never a reason to silently alter that schedule.
pub fn int4_faithfulness_document_corpus_requirements(width: usize, task: TaskBudget,
    planning: JudgeLimits, mapping: Int4FaithfulnessMapLimits, limits: CorpusLimits,
    cohort_chunks: Option<usize>) -> Result<Int4FaithfulnessMapMemoryRequirement, HostedError> {
    int4_faithfulness_document_corpus_requirements_with_max_heads(width, task, planning,
        mapping, limits, MAX_FAITHFULNESS_MAP_HEADS, cohort_chunks)
}

/// Quote a corpus whose sealed planner explicitly bounds complete evidence
/// heads per source chunk. The planner checks that its fixed policy fits this
/// declaration before execution; the quote cannot grant a wider policy or
/// omit any evidence. The legacy helper retains its original 32-head bound.
pub fn int4_faithfulness_document_corpus_requirements_with_max_heads(width: usize, task: TaskBudget,
    planning: JudgeLimits, mapping: Int4FaithfulnessMapLimits, limits: CorpusLimits,
    max_heads: usize, cohort_chunks: Option<usize>) -> Result<Int4FaithfulnessMapMemoryRequirement, HostedError> {
    let transport = limits.reservation_bytes()?;
    let max_result_bytes = document_output_bytes(mapping, cohort_chunks)?;
    if max_result_bytes > limits.transport.max_output_line_bytes as u64 {
        return Err(HostedError::Limits("complete faithfulness document output reservation"));
    }
    let memory = map_memory(limits);
    let mut required = match cohort_chunks {
        None => memory.requirements_with_max_heads(width, task, planning, mapping, max_heads),
        Some(chunks) => memory.cohort_requirements(width, task, planning, mapping,
            max_heads, chunks).map(|required| required.memory),
    }.map_err(map_error)?;
    let metadata = (mapping.chunks.max_chunks as u64).checked_mul(FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES)
        .ok_or(HostedError::Limits("faithfulness document metadata arithmetic"))?;
    required.input_bytes = sum(&[required.input_bytes, transport, metadata])?;
    sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
        required.output_reservation_bytes])?;
    Ok(required)
}
pub(super) fn map_memory(limits: CorpusLimits) -> Int4FaithfulnessMapMemoryLimits {
    Int4FaithfulnessMapMemoryLimits { context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes }
}
pub(super) fn document_output_bytes(mapping: Int4FaithfulnessMapLimits, cohort_chunks: Option<usize>)
    -> Result<u64, HostedError> {
    mapping.max_result_bytes.checked_add(if cohort_chunks.is_some() { 4096 } else { 0 })
        .ok_or(HostedError::Limits("faithfulness document output arithmetic"))
}

struct StreamInput<R, W> { planner: Option<Int4FaithfulnessDocumentBatchPlanner>, reader: R, writer: W }

struct DeferredFaithfulnessStream<R, W> {
    completion: stream::deferred::DeferredCompletion,
    input: Charged<StreamInput<R, W>>,
}

impl NlpEngine {
    /// Audit complete original sources and per-row exact claims under one
    /// resident model, deadline and five-axis invocation work ledger. A result
    /// includes every chunk and original-coordinate witness before it can be
    /// published. Flush boundaries never replenish work or discard context.
    ///
    /// The planner and supplied reader/writer move into the physical owned
    /// invocation, and are destroyed before completion is reported. IO remains
    /// cooperative; no per-item runtime, model reload or detached worker exists.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_faithfulness_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4FaithfulnessDocumentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_faithfulness_documents(model, planner, limits, kernel, None,
            reader, writer, cancellation)
    }

    /// Use one persistent numerical row team across the complete stream:
    /// every original source, exact claim, chunk and evidence scoring head.
    /// The existing evidence finalizer and five-axis work ledger remain in
    /// force. Terminal completion follows physical join under the original
    /// deadline. Explicit native chunk cohorts cannot be combined.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_faithfulness_documents_with_threads<R, W>(&self, model: &ResidentInt4,
        planner: Int4FaithfulnessDocumentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        self.run_int4_faithfulness_documents(model, planner, limits, kernel, Some(compute_threads),
            reader, writer, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_faithfulness_documents<R, W>(&self, model: &ResidentInt4,
        planner: Int4FaithfulnessDocumentBatchPlanner, limits: CorpusLimits, kernel: PackedInt4Kernel,
        compute_threads: Option<usize>, reader: R, writer: W, cancellation: CancellationToken)
        -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        judge_task::check_model(planner.base_execution_identity(), model.model_identity()).map_err(judge_error)?;
        if planner.group_width() != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        let mapping = planner.mapping_limits();
        crate::batch::judge::grouped_int4::validate_limits(planner.task_budget(), planner.planning_limits(),
            mapping, planner.max_run_work()).map_err(HostedError::BatchSetup)?;
        let cohort_chunks = planner.cohort_chunks();
        check_team_schedule(compute_threads, cohort_chunks)?;
        let required = int4_faithfulness_document_corpus_requirements_with_max_heads(
            planner.group_width(), planner.task_budget(), planner.planning_limits(), mapping,
            limits, planner.max_heads(), cohort_chunks)?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, planner.group_width())).transpose()?.unwrap_or(0);
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes,
            required.output_reservation_bytes, team_scratch])?;
        let capacity = DocumentCapacity { context: limits.native.context_tokens, kv_bytes: required.native.kv_bytes,
            mapping, cohort_chunks, max_heads: planner.max_heads(),
            max_result_bytes: planner.max_result_bytes(), task_budget: planner.task_budget() };
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
                // The declared peak remains charged while every exact ragged
                // window is allocated, executed, cleared and destroyed.
                let kv = kv.commit()?; let scratch = scratch.commit()?;
                let result = {
                    let mut processor = NativeInt4FaithfulnessDocumentBatch::new_cohorts(planner,
                        &model.inner.loaded.value, kernel, map_memory(limits), admission)
                        .map_err(HostedError::BatchSetup)?;
                    stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)
                };
                drop(scratch); drop(kv);
                result
            } else {
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                        native_budget(required.native)).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
                })?;
                let result = {
                    let mut processor = NativeInt4FaithfulnessDocumentBatch::new(planner, &mut engine.value, admission)
                        .map_err(HostedError::BatchSetup)?;
                    stream::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)
                };
                drop(engine);
                result
            };
            drop(input); // Retained private IO precedes physical completion.
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
                    StrictInt4Engine::new(weights, limits.native.context_tokens,
                        native_budget(required.native)).map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let completion = {
                    let mut processor = NativeInt4FaithfulnessDocumentBatch::new(planner,
                        &mut engine.value, admission).map_err(HostedError::BatchSetup)?;
                    stream::deferred::run_ndjson_deferred(&mut input.value.reader, &mut input.value.writer,
                        &mut processor, limits.transport, control).map_err(HostedError::Batch)?
                };
                drop(engine); drop(lease);
                Ok(DeferredFaithfulnessStream { completion, input })
            }, |pending, control| pending.publish(control)),
        }
    }
}

fn check_team_schedule(threads: Option<usize>, cohort_chunks: Option<usize>) -> Result<(), HostedError> {
    if threads.is_some() && cohort_chunks.is_some() {
        return Err(HostedError::Limits("INT4 faithfulness corpus team cannot combine with native chunk cohorts"));
    }
    Ok(())
}

impl<R, W: Write> DeferredFaithfulnessStream<R, W> {
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
    pub(super) mapping: Int4FaithfulnessMapLimits,
    pub(super) cohort_chunks: Option<usize>,
    pub(super) max_heads: usize,
    pub(super) max_result_bytes: u64,
    pub(super) task_budget: TaskBudget,
}
pub(super) struct DocumentAdmission<'a> {
    pub(super) lease: &'a EngineLease,
    pub(super) model: &'a Int4ModelIdentity,
    pub(super) capacity: DocumentCapacity,
    pub(super) output_bytes: u64,
}
impl Int4FaithfulnessDocumentBatchAdmission for DocumentAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4FaithfulnessDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Pending), BatchItemFailure> {
        check_request(self.model.group_width(), self.capacity, &request)?;
        for identity in request.identities {
            judge_task::check_model(identity, self.model)
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
fn check_request(width: usize, capacity: DocumentCapacity, request: &Int4FaithfulnessDocumentAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let c = capacity;
    let chunks = request.identities.len();
    if request.group_width != width || request.kv_reservation_bytes != c.kv_bytes
        || request.task_budget != c.task_budget || request.task_budget.validate().is_err()
        || chunks == 0 || chunks > c.mapping.chunks.max_chunks
        || request.required_context == 0 || request.required_context > c.context
        || !(1..=MAX_FAITHFULNESS_MAP_HEADS).contains(&c.max_heads)
        || !(1..=c.max_heads).contains(&request.max_head_count)
        || request.head_count < chunks || request.head_count < request.max_head_count
        || chunks.checked_mul(request.max_head_count).is_none_or(|heads| request.head_count > heads)
        || request.cohort_chunks != c.cohort_chunks {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let row = Int4MemoryRequirement::for_context(c.context, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let physical = if let Some(chunks) = c.cohort_chunks {
        if chunks == 0 || chunks > c.mapping.chunks.max_chunks || chunks > MAX_INT4_SEQUENCES {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let heads = chunks.checked_mul(c.max_heads)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?.min(MAX_INT4_SEQUENCES);
        let contexts = [c.context; MAX_INT4_SEQUENCES];
        if row.kv_bytes.checked_mul(request.max_head_count as u64)
            .is_none_or(|bytes| bytes > request.task_budget.max_kv_bytes) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Int4MemoryRequirement::for_cohort(&contexts[..heads], width)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?
    } else { row };
    if physical.kv_bytes != c.kv_bytes || row.kv_bytes > request.task_budget.max_kv_bytes {
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
        if identity.task_spec != "judge-v1" { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    }
    Ok(())
}

#[cfg(test)] mod tests;
